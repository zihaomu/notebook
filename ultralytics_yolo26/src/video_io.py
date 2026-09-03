"""GPU video I/O — hardware decode (rocDecode/VCN) and encode (VA-API/VCN).

Decode: rocDecode via its rocPyDecode Python bindings. Decoded frames stay in
VRAM and are exposed as PyTorch GPU tensors through DLPack — no host copy on the
decode path. This feeds the GPU-resident preprocess → detect chain directly.

Encode: the production path maps linear VAAPI NV12 surfaces into HIP, performs
RGB conversion and overlay on the GPU, then submits the same surfaces to
h264_vaapi. A host BGR pipe remains available for CPU/VLM fallback paths.

Both paths fall back to OpenCV (cv2.VideoCapture / cv2.VideoWriter, CPU FFmpeg)
when the GPU backend is unavailable, so the pipeline always runs.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

# rocDecode bindings live under /opt/rocm/lib
if "/opt/rocm/lib" not in sys.path:
    sys.path.insert(0, "/opt/rocm/lib")


# ---------------------------------------------------------------------------
# GPU hardware decode — rocDecode
# ---------------------------------------------------------------------------

class RocDecodeReader:
    """Hardware video decoder built on rocDecode / VCN.

    Yields decoded frames as PyTorch GPU tensors (H, W, 3) uint8, RGB, on the
    current HIP device — ready to feed a GPU-resident preprocessing stage with
    no host round-trip.
    """

    def __init__(self, path, device_id=0):
        import torch
        import pyRocVideoDecode.demuxer as dmx
        import pyRocVideoDecode.decoder as dec
        from pyRocVideoDecode.types import OUT_SURFACE_MEM_DEV_COPIED

        self._torch = torch
        self._demuxer = dmx.demuxer(path)
        codec_id = dec.GetRocDecCodecID(self._demuxer.GetCodecId())
        self._decoder = dec.decoder(
            codec_id,
            device_id=device_id,
            mem_type=OUT_SURFACE_MEM_DEV_COPIED,
            b_force_zero_latency=False,
        )
        self._pending = []          # decoded surfaces not yet returned
        self._eos = False
        self.width = self._decoder.GetWidth()
        self.height = self._decoder.GetHeight()

    def read_gpu(self):
        """Return (ok, rgb_gpu_tensor). Tensor is (H, W, 3) uint8 on GPU.

        The tensor is a view into the decoder's surface; clone it if you need it
        to outlive the next read_gpu() call.
        """
        torch = self._torch
        while not self._pending and not self._eos:
            packet = self._demuxer.DemuxFrame()
            n = self._decoder.DecodeFrame(packet)
            for _ in range(n):
                if self._decoder.GetFrameRgb(packet, rgb_format=3) == -1:
                    continue
                raw = torch.from_dlpack(packet.ext_buf[0])
                height, width = raw.shape[:2]
                if raw.stride() == (width * 3, 1, 0):
                    # rocPyDecode 0.8.0 reports an invalid channel stride and
                    # declares storage two channels short on the final row.
                    visible = raw.as_strided(
                        (height - 1, width, 3), (width * 3, 3, 1)
                    ).clone()
                    t = torch.cat((visible, visible[-1:].clone()), dim=0)
                else:
                    t = raw.clone()
                self._pending.append(t)
                self._decoder.ReleaseFrame(packet)
            if packet.bitstream_size <= 0:
                self._eos = True
        if self._pending:
            return True, self._pending.pop(0)
        return False, None

    def release(self):
        self._demuxer = None
        self._decoder = None


def make_reader(path, device_id=0, prefer_gpu=True):
    """Return (reader, kind). kind is 'rocdecode' or 'opencv'.

    Tries rocDecode first; on any failure falls back to cv2.VideoCapture.
    """
    if prefer_gpu:
        try:
            r = RocDecodeReader(path, device_id=device_id)
            return r, "rocdecode"
        except Exception as e:
            print(f"[video_io] rocDecode unavailable ({e}); using OpenCV decode")
    import cv2
    cap = cv2.VideoCapture(path)
    return cap, "opencv"


# ---------------------------------------------------------------------------
# GPU hardware encode — ffmpeg VA-API
# ---------------------------------------------------------------------------

class VaapiWriter:
    """Feed ffmpeg VA-API from a bounded worker queue to overlap GPU stages."""

    def __init__(self, path, width, height, fps, device=None):
        self._ffmpeg = shutil.which("ffmpeg")
        if not self._ffmpeg:
            raise RuntimeError("ffmpeg not found")
        self.device = device or os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128")
        cmd = [
            self._ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
            "-i", "-",
            "-vaapi_device", self.device,
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-qp", "24",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        depth = int(os.environ.get("VIDEO_ENCODE_QUEUE_DEPTH", "8"))
        if depth < 1:
            raise ValueError("VIDEO_ENCODE_QUEUE_DEPTH must be at least 1")
        self._queue = queue.Queue(maxsize=depth)
        self._error = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._write_loop, name="vaapi-writer", daemon=True
        )
        self._worker.start()

    def _write_loop(self):
        try:
            while True:
                payload = self._queue.get()
                try:
                    if payload is None:
                        return
                    assert self._proc.stdin is not None
                    frame = np.ascontiguousarray(payload)
                    self._proc.stdin.write(memoryview(frame).cast("B"))
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._error = error

    def _put(self, payload):
        while True:
            if self._error is not None:
                raise RuntimeError("VA-API writer worker failed") from self._error
            try:
                self._queue.put(payload, timeout=0.1)
                return
            except queue.Full:
                continue

    def write(self, frame_bgr):
        if self._closed:
            raise RuntimeError("VA-API writer is closed")
        self._put(np.ascontiguousarray(frame_bgr))

    def release(self):
        if self._closed:
            return
        self._closed = True
        self._put(None)
        self._queue.join()
        self._worker.join()
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            stderr = (
                self._proc.stderr.read().decode("utf-8", "ignore")
                if self._proc.stderr else ""
            )
            self._proc.wait()
            return_code = self._proc.returncode
            self._proc = None
            if self._error is not None:
                raise RuntimeError("VA-API writer worker failed") from self._error
            if return_code:
                raise RuntimeError(
                    f"ffmpeg VA-API exited with {return_code}: {stderr.strip()[:500]}"
                )


class GpuDirectVaapiWriter:
    """Encode resident RGB tensors on a dedicated HIP stream and worker."""

    is_gpu_direct = True

    def __init__(self, path, width, height, fps, device=None, qp=24):
        import torch
        from hip_vaapi_bridge import HipVaapiEncoder

        self.device = device or os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128")
        self._torch = torch
        self._torch_device = torch.cuda.current_device()
        self._stream = torch.cuda.Stream(device=self._torch_device)
        self._encoder = HipVaapiEncoder(
            str(path), self.device, int(width), int(height), float(fps), int(qp)
        )
        depth = int(os.environ.get("GPU_DIRECT_ENCODE_QUEUE_DEPTH", "3"))
        if depth < 1:
            raise ValueError("GPU_DIRECT_ENCODE_QUEUE_DEPTH must be at least 1")
        self._queue = queue.Queue(maxsize=depth)
        self._queue_depth = depth
        self._error = None
        self._closed = False
        self._submitted_frames = 0
        self._encoded_frames = 0
        self._worker_seconds = 0.0
        self._encoder_info = None
        self._worker = threading.Thread(
            target=self._write_loop, name="gpu-direct-vaapi-writer", daemon=True
        )
        self._worker.start()

    def _write_loop(self):
        torch = self._torch
        try:
            torch.cuda.set_device(self._torch_device)
            while True:
                payload = self._queue.get()
                try:
                    if payload is None:
                        return
                    rgb_gpu, ready_event, detections, labels, status_lines = payload
                    self._stream.wait_event(ready_event)
                    started = time.perf_counter()
                    self._encoder.write(
                        rgb_gpu.data_ptr(),
                        int(rgb_gpu.stride(0)),
                        detections,
                        labels,
                        self._stream.cuda_stream,
                        status_lines,
                    )
                    self._worker_seconds += time.perf_counter() - started
                    self._encoded_frames += 1
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._error = error

    def _put(self, payload):
        while True:
            if self._error is not None:
                raise RuntimeError("GPU direct VA-API writer worker failed") from self._error
            try:
                self._queue.put(payload, timeout=0.1)
                return
            except queue.Full:
                continue

    def write_gpu(self, rgb_gpu, detections, names, status_lines=()):
        torch = self._torch
        if self._closed:
            raise RuntimeError("GPU direct VA-API writer is closed")
        if rgb_gpu.device.type != "cuda" or rgb_gpu.dtype != torch.uint8:
            raise ValueError(
                f"Expected a resident uint8 RGB tensor, got {rgb_gpu.device}/{rgb_gpu.dtype}"
            )
        if rgb_gpu.ndim != 3 or rgb_gpu.shape[2] != 3 or not rgb_gpu.is_contiguous():
            raise ValueError(
                f"Expected contiguous HWC RGB tensor, got {tuple(rgb_gpu.shape)}"
            )
        labels = [
            f"{names[int(item[5])]} {item[4]:.2f}"
            for item in detections
        ]
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(rgb_gpu.device))
        self._put((
            rgb_gpu,
            ready_event,
            detections,
            labels,
            list(status_lines),
        ))
        self._submitted_frames += 1

    def write(self, frame_bgr):
        raise TypeError("GpuDirectVaapiWriter requires write_gpu()")

    def release(self):
        if self._closed:
            return
        self._closed = True
        if self._error is None:
            try:
                self._put(None)
            except RuntimeError:
                pass
        self._worker.join()
        self._encoder_info = dict(self._encoder.info())
        self._encoder.close()
        if self._error is not None:
            raise RuntimeError("GPU direct VA-API writer worker failed") from self._error
        if self._encoded_frames != self._submitted_frames:
            raise RuntimeError(
                f"GPU direct writer encoded {self._encoded_frames} of "
                f"{self._submitted_frames} submitted frames"
            )

    def info(self):
        result = dict(self._encoder_info or self._encoder.info())
        result.update({
            "queue_depth": self._queue_depth,
            "submitted_frames": self._submitted_frames,
            "encoded_frames": self._encoded_frames,
            "worker_seconds": self._worker_seconds,
        })
        return result


def make_gpu_writer(path, width, height, fps, device=None):
    """Create the HIP overlay plus DRM PRIME VAAPI writer."""
    return GpuDirectVaapiWriter(path, width, height, fps, device=device), "vaapi-direct"


def make_writer(path, width, height, fps, prefer_gpu=True, device=None):
    """Return (writer, kind). kind is 'vaapi' or 'opencv'.

    Tries VA-API (ffmpeg) first; on any failure falls back to cv2.VideoWriter.
    The returned object always exposes .write(frame_bgr) and .release().
    """
    if prefer_gpu:
        try:
            w = VaapiWriter(path, width, height, fps, device=device)
            return w, "vaapi"
        except Exception as e:
            print(f"[video_io] VA-API encode unavailable ({e}); using OpenCV encode")
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return _OpenCVWriter(writer), "opencv"


class _OpenCVWriter:
    """Thin adapter so cv2.VideoWriter matches the VaapiWriter interface."""

    def __init__(self, writer):
        self._writer = writer

    def write(self, frame_bgr):
        self._writer.write(frame_bgr)

    def release(self):
        self._writer.release()
