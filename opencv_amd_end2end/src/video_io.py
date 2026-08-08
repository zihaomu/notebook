"""GPU video I/O — hardware decode (rocDecode/VCN) and encode (VA-API/VCN).

Decode: rocDecode via its rocPyDecode Python bindings. Decoded frames stay in
VRAM and are exposed as PyTorch GPU tensors through DLPack — no host copy on the
decode path. This feeds the GPU-resident preprocess → detect chain directly.

Encode: ffmpeg's VA-API encoder (h264_vaapi) driven over a subprocess pipe. The
AMD VCN engine does the encoding; we push annotated BGR frames to ffmpeg stdin.

Both paths fall back to OpenCV (cv2.VideoCapture / cv2.VideoWriter, CPU FFmpeg)
when the GPU backend is unavailable, so the pipeline always runs.
"""

import shutil
import subprocess
import sys

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
    """Hardware video encoder via ffmpeg's h264_vaapi (AMD VCN).

    Accepts annotated BGR frames (numpy uint8) and pipes them to ffmpeg, which
    uploads to a VA-API surface and encodes on the VCN engine.
    """

    def __init__(self, path, width, height, fps, device="/dev/dri/renderD128"):
        self._ffmpeg = shutil.which("ffmpeg")
        if not self._ffmpeg:
            raise RuntimeError("ffmpeg not found")
        cmd = [
            self._ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
            "-i", "-",
            "-vaapi_device", device,
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-qp", "24",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr):
        self._proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def release(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            err = self._proc.stderr.read().decode("utf-8", "ignore") if self._proc.stderr else ""
            self._proc.wait()
            if self._proc.returncode not in (0, None) and err:
                print(f"[video_io] ffmpeg VA-API encoder: {err.strip()[:300]}")
            self._proc = None


def make_writer(path, width, height, fps, prefer_gpu=True):
    """Return (writer, kind). kind is 'vaapi' or 'opencv'.

    Tries VA-API (ffmpeg) first; on any failure falls back to cv2.VideoWriter.
    The returned object always exposes .write(frame_bgr) and .release().
    """
    if prefer_gpu:
        try:
            w = VaapiWriter(path, width, height, fps)
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
