"""GPU-resident ROI preprocessing + HIP IPC export for the zero-copy VLM path.

The pipeline keeps each ROI on the GPU, runs the Qwen3-VL image preprocess
(smart_resize -> aspect-fit bilinear with align_corners -> center pad ->
normalize) entirely on the GPU via cv::cuda (HIP) — the same OpenCV GPU backend
the rest of the pipeline uses — then exports a HIP IPC handle so the llama.cpp
server can map the same VRAM and feed the vision encoder directly: no JPEG
encode, no pixels over HTTP, no host round-trip.

Using cv::cuda (not torch) keeps the whole pipeline on one GPU stack and avoids
torch's caching-allocator contending for VRAM with the co-located llama-server.
It relies on the `align_corners=True` option added to cv::cuda::resize (which
routes to a truncating-linear kernel), making the preprocess numerically
identical (verified element-wise, zero diff) to llama.cpp's CPU preprocess, so
the server skips its own resize/normalize (multimodal_data "preprocessed": true).
"""

import ctypes

import cv2
import numpy as np

# --- Qwen3-VL preprocess params (from mmproj GGUF clip.vision.* + clip.cpp) ---
PATCH_SIZE = 16
N_MERGE = 2
ALIGN = PATCH_SIZE * N_MERGE                          # 32
_PATCH_AREA = PATCH_SIZE * PATCH_SIZE * N_MERGE * N_MERGE  # 1024
MIN_PIXELS = 8 * _PATCH_AREA                          # 8192
MAX_PIXELS = 4096 * _PATCH_AREA                       # 4194304

# --- HIP runtime via ctypes (export IPC handle for a dedicated GPU buffer) ---
_HIP = ctypes.CDLL("libamdhip64.so")
for _fn in ("hipMalloc", "hipFree", "hipMemcpy", "hipMemcpy2D", "hipIpcGetMemHandle", "hipDeviceSynchronize"):
    getattr(_HIP, _fn).restype = ctypes.c_int
_HIP.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_HIP.hipFree.argtypes = [ctypes.c_void_p]
_HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_HIP.hipMemcpy2D.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
_HIP_MEMCPY_D2D = 3


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


_HIP.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: hip error {rc}")


def _smart_resize(w, h):
    """Replicate llama.cpp calc_size_preserved_ratio(min_pixels, max_pixels).
    Uses floor(x+0.5) to match C++ std::round (round-half-away-from-zero)."""
    def rnd(x): return int(np.floor(x / ALIGN + 0.5)) * ALIGN
    def cel(x): return int(np.ceil(x / ALIGN)) * ALIGN
    def flr(x): return int(np.floor(x / ALIGN)) * ALIGN
    h_bar = max(ALIGN, rnd(h))
    w_bar = max(ALIGN, rnd(w))
    if h_bar * w_bar > MAX_PIXELS:
        beta = np.sqrt((h * w) / MAX_PIXELS)
        h_bar = max(ALIGN, flr(h / beta))
        w_bar = max(ALIGN, flr(w / beta))
    elif h_bar * w_bar < MIN_PIXELS:
        beta = np.sqrt(MIN_PIXELS / (h * w))
        h_bar = cel(h * beta)
        w_bar = cel(w * beta)
    return w_bar, h_bar


def preprocess_roi_gpu(roi_rgb):
    """Preprocess an RGB ROI on the GPU with cv::cuda (HIP).

    Args:
        roi_rgb: a cv2.cuda_GpuMat (CV_8UC3, RGB) OR a torch.Tensor (H,W,3) uint8
                 CUDA RGB. A torch tensor is wrapped zero-copy as a GpuMat.

    Returns:
        (nx, ny, gpu_f32) where gpu_f32 is a cv2.cuda_GpuMat of shape (ny, nx)
        CV_32FC3 holding interleaved-RGB normalized pixel_values, numerically
        identical to llama.cpp's Qwen3-VL preprocess.
    """
    if isinstance(roi_rgb, cv2.cuda_GpuMat):
        g = roi_rgb
        w, h = g.size()  # (cols, rows)
    else:
        # torch.Tensor (H,W,3) uint8 CUDA -> zero-copy GpuMat over its memory
        roi_rgb = roi_rgb.contiguous()
        h, w = int(roi_rgb.shape[0]), int(roi_rgb.shape[1])
        g = cv2.cuda_GpuMat.fromDevicePointer(roi_rgb.data_ptr(), h, w, cv2.CV_8UC3, w * 3)

    nx, ny = _smart_resize(w, h)
    scale = min(nx / w, ny / h)
    new_w = min(int(np.ceil(w * scale)), nx)
    new_h = min(int(np.ceil(h * scale)), ny)

    # align_corners=True routes to the truncating-linear kernel -> bit-exact
    resized = cv2.cuda.resize(g, (new_w, new_h), interpolation=cv2.INTER_LINEAR,
                              align_corners=True)
    off_x = (nx - new_w) // 2
    off_y = (ny - new_h) // 2
    padded = cv2.cuda.copyMakeBorder(resized, off_y, ny - new_h - off_y,
                                     off_x, nx - new_w - off_x,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gpu_f32 = cv2.cuda_GpuMat(padded.size(), cv2.CV_32FC3)
    padded.convertTo(cv2.CV_32FC3, gpu_f32, alpha=1.0 / 127.5, beta=-1.0)  # (v-127.5)/127.5
    return nx, ny, gpu_f32


class IpcBuffer:
    """A dedicated hipMalloc'd GPU buffer holding tightly-packed f32 pixels, with
    its IPC handle.

    We copy the preprocess output into a standalone hipMalloc allocation rather
    than exporting the GpuMat's pointer directly, for two reasons: (1) a GpuMat
    may be row-padded (step > width*elem), but the server expects a tightly
    packed nx*ny*3 f32 buffer; (2) a dedicated allocation is guaranteed to be at
    offset 0 so the exported IPC handle maps to exactly this buffer.

    Must be kept alive until the server has finished mapping and reading it
    (i.e. until the HTTP request returns).
    """

    def __init__(self, gpu_f32):
        # gpu_f32: cv2.cuda_GpuMat, CV_32FC3, (ny, nx). Copy row-by-row into a
        # packed device buffer (handles GpuMat step/padding via hipMemcpy2D).
        w, h = gpu_f32.size()          # (cols, nx), (rows, ny)
        row_bytes = w * 3 * 4          # 3 channels, f32
        self.nbytes = row_bytes * h
        src_ptr = ctypes.c_void_p(gpu_f32.cudaPtr())
        src_step = gpu_f32.step
        self._dptr = ctypes.c_void_p()
        _ck(_HIP.hipMalloc(ctypes.byref(self._dptr), ctypes.c_size_t(self.nbytes)), "hipMalloc")
        _ck(_HIP.hipMemcpy2D(self._dptr, ctypes.c_size_t(row_bytes),
                             src_ptr, ctypes.c_size_t(src_step),
                             ctypes.c_size_t(row_bytes), ctypes.c_size_t(h),
                             _HIP_MEMCPY_D2D), "hipMemcpy2D D2D")
        _ck(_HIP.hipDeviceSynchronize(), "hipDeviceSynchronize")
        hh = _Handle()
        _ck(_HIP.hipIpcGetMemHandle(ctypes.byref(hh), self._dptr), "hipIpcGetMemHandle")
        self.handle = ctypes.string_at(ctypes.byref(hh), 64)

    def free(self):
        if self._dptr:
            _HIP.hipFree(self._dptr)
            self._dptr = ctypes.c_void_p()

    def __del__(self):
        self.free()
