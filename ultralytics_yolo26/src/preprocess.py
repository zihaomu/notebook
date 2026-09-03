"""GPU-accelerated preprocessing using cv::cuda (HIP backend on AMD GPUs)."""

import cv2
import numpy as np

import config


def letterbox_gpu(gpu_frame, target_size):
    """Resize with aspect ratio preservation and padding on GPU using warpAffine.

    Uses a single warpAffine call to combine resize + translate (padding) into one
    GPU kernel — more efficient than separate resize + copyMakeBorder.

    Returns (gpu_padded, scale, pad_w, pad_h).
    """
    h, w = gpu_frame.size()[::-1]  # GpuMat.size() returns (cols, rows)
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w, pad_h = (tw - new_w) // 2, (th - new_h) // 2

    # Affine matrix: scale + translate to center
    M = np.array([
        [scale, 0, pad_w],
        [0, scale, pad_h],
    ], dtype=np.float64)

    gpu_padded = cv2.cuda.warpAffine(
        gpu_frame, M, (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )

    return gpu_padded, scale, pad_w, pad_h


def preprocess_frame(frame, stream=None):
    """Full GPU preprocessing pipeline for YOLO input.

    Args:
        frame: BGR numpy array (H, W, 3) uint8

    Returns:
        blob: numpy array (1, 3, 640, 640) float32, ready for MIGraphX
        scale, pad_w, pad_h: letterbox parameters for coordinate mapping
    """
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    gpu_padded, scale, pad_w, pad_h = letterbox_gpu(gpu_frame, config.INPUT_SIZE)

    gpu_rgb = cv2.cuda.cvtColor(gpu_padded, cv2.COLOR_BGR2RGB)

    gpu_float = cv2.cuda_GpuMat(gpu_rgb.size(), cv2.CV_32FC3)
    gpu_rgb.convertTo(cv2.CV_32FC3, gpu_float, alpha=1.0 / 255.0)

    blob_hwc = gpu_float.download()
    blob = blob_hwc.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    return blob.astype(np.float32), scale, pad_w, pad_h


class GPUPreprocessor:
    """Reuse fixed GPU buffers for the rocDecode-to-YOLO preprocessing path."""

    def __init__(self, target_size, device="cuda:0"):
        import torch

        self.target_size = tuple(target_size)
        self._uint8 = torch.uint8
        tw, th = self.target_size
        self.padded = cv2.cuda_GpuMat(th, tw, cv2.CV_8UC3)
        self.float_hwc = cv2.cuda_GpuMat(th, tw, cv2.CV_32FC3)
        self.hwc = torch.empty((th, tw, 3), dtype=torch.float32, device=device)
        self.blob = torch.empty((1, 3, th, tw), dtype=torch.float32, device=device)
        self.hwc_view = cv2.cuda_GpuMat.fromDevicePointer(
            self.hwc.data_ptr(), th, tw, cv2.CV_32FC3, tw * 3 * 4
        )
        self._source_size = None
        self._matrix = None
        self._letterbox = None

    def _geometry(self, width, height):
        if self._source_size != (width, height):
            tw, th = self.target_size
            scale = min(tw / width, th / height)
            new_w, new_h = int(width * scale), int(height * scale)
            pad_w, pad_h = (tw - new_w) // 2, (th - new_h) // 2
            self._matrix = np.array(
                [[scale, 0, pad_w], [0, scale, pad_h]], dtype=np.float64
            )
            self._letterbox = (scale, pad_w, pad_h)
            self._source_size = (width, height)
        return self._matrix, self._letterbox

    def process(self, rgb_gpu, stream=None):
        """Return the stable BCHW buffer filled from a resident RGB tensor."""
        if not rgb_gpu.is_contiguous():
            raise ValueError("rocDecode RGB tensor must be contiguous")
        if rgb_gpu.dtype != self._uint8:
            raise ValueError(f"Expected uint8 RGB input, got {rgb_gpu.dtype}")

        height, width = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
        gpu_rgb = cv2.cuda_GpuMat.fromDevicePointer(
            rgb_gpu.data_ptr(), height, width, cv2.CV_8UC3, width * 3
        )
        matrix, letterbox = self._geometry(width, height)
        cv2.cuda.warpAffine(
            gpu_rgb,
            matrix,
            self.target_size,
            self.padded,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        self.padded.convertTo(
            cv2.CV_32FC3, self.float_hwc, alpha=1.0 / 255.0
        )
        self.float_hwc.copyTo(self.hwc_view)
        self.blob.copy_(self.hwc.permute(2, 0, 1).unsqueeze(0))
        return self.blob, *letterbox

    def pointer_info(self):
        return {
            "hwc_pointer": self.hwc.data_ptr(),
            "blob_pointer": self.blob.data_ptr(),
            "padded_pointer": int(self.padded.cudaPtr()),
            "float_hwc_pointer": int(self.float_hwc.cudaPtr()),
        }


def preprocess_frame_gpu_resident(rgb_gpu, target_size, stream=None, workspace=None):
    """Preprocess a rocDecode tensor with OpenCV HIP into a GPU YOLO tensor."""
    processor = workspace or GPUPreprocessor(target_size, device=rgb_gpu.device)
    return processor.process(rgb_gpu, stream=stream)


def preprocess_frame_cpu(frame):
    """CPU fallback preprocessing (for comparison/debugging)."""
    h, w = frame.shape[:2]
    tw, th = config.INPUT_SIZE
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w, pad_h = (tw - new_w) // 2, (th - new_h) // 2

    M = np.array([
        [scale, 0, pad_w],
        [0, scale, pad_h],
    ], dtype=np.float64)

    padded = cv2.warpAffine(
        frame, M, (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    return blob, scale, pad_w, pad_h
