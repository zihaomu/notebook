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


def preprocess_frame_gpu_resident(rgb_gpu, target_size, stream=None):
    """GPU-resident preprocessing with cv::cuda (HIP) — no host copy.

    Takes the (H, W, 3) uint8 RGB GPU tensor produced by the hardware decoder
    (rocDecode) and runs the same cv::cuda warpAffine/cvtColor/convertTo path
    used for CPU-decoded frames — but the input is wrapped as a GpuMat directly
    over the decoder's GPU memory (zero copy) via GpuMat.fromDevicePointer.

    The output blob is produced as a GPU torch tensor (1, 3, th, tw) float32 so
    it flows zero-copy into the MIGraphX detector.

    Args:
        rgb_gpu: torch.Tensor (H, W, 3) uint8 on CUDA, RGB channel order.
        target_size: (tw, th)

    Returns:
        blob_gpu: torch.Tensor (1, 3, th, tw) float32 on CUDA
        scale, pad_w, pad_h: letterbox parameters for coordinate mapping
    """
    import torch

    rgb_gpu = rgb_gpu.contiguous()
    h, w = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
    step = w * 3  # uint8, 3 channels, tightly packed

    # Wrap the decoder's GPU memory as a GpuMat (zero copy, no ownership)
    gpu_rgb_in = cv2.cuda_GpuMat.fromDevicePointer(
        rgb_gpu.data_ptr(), h, w, cv2.CV_8UC3, step
    )

    # letterbox (RGB in, RGB out — decoder already gives RGB)
    gpu_padded, scale, pad_w, pad_h = letterbox_gpu(gpu_rgb_in, target_size)

    # normalize to float32 [0,1]
    tw, th = target_size
    gpu_float = cv2.cuda_GpuMat(gpu_padded.size(), cv2.CV_32FC3)
    gpu_padded.convertTo(cv2.CV_32FC3, gpu_float, alpha=1.0 / 255.0)

    # HWC float32 GpuMat -> wrap as torch tensor (zero copy), then to CHW blob
    hwc = torch.empty((th, tw, 3), dtype=torch.float32, device=rgb_gpu.device)
    dst = cv2.cuda_GpuMat.fromDevicePointer(hwc.data_ptr(), th, tw, cv2.CV_32FC3, tw * 3 * 4)
    gpu_float.copyTo(dst)
    blob = hwc.permute(2, 0, 1).unsqueeze(0).contiguous()

    return blob, scale, pad_w, pad_h


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
