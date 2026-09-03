#!/opt/venv/bin/python3
"""Smoke-test the Ultralytics YOLO26x Radeon workshop image."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    workspace = Path(
        os.environ.get("ULTRALYTICS_YOLO26_ROOT", "/workspace")
    ).resolve()
    if workspace != Path("/workspace"):
        raise RuntimeError(f"Expected workspace /workspace, got {workspace}")

    required_paths = [
        workspace / "scripts/notebook_env.py",
        workspace / "tests/test_ultralytics_migraphx_backend.py",
        workspace / "src/video_io.py",
        workspace / "data/sidewalk.mp4",
        workspace / "models/yolo26x.pt",
        workspace / "models/yolo26x.onnx",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image/package paths: {missing}")

    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(workspace / "src"))

    import cv2
    import jupyterlab
    import numpy as np
    import onnxruntime as ort
    import torch
    import ultralytics
    from hip_vaapi_bridge import HipVaapiEncoder
    from tests.test_ultralytics_migraphx_backend import validate as validate_backend
    from video_io import RocDecodeReader

    if ultralytics.__version__ != "8.4.75":
        raise RuntimeError(f"Unexpected Ultralytics version: {ultralytics.__version__}")
    if ort.__version__ != "1.24.2":
        raise RuntimeError(f"Unexpected ONNX Runtime version: {ort.__version__}")
    if np.__version__ != "1.26.4":
        raise RuntimeError(f"Unexpected NumPy version: {np.__version__}")
    if "MIGraphXExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"MIGraphX provider unavailable: {ort.get_available_providers()}")
    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        raise RuntimeError("OpenCV HIP cannot see a GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a ROCm GPU")
    if not hasattr(cv2.cuda_GpuMat, "fromDevicePointer"):
        raise RuntimeError("OpenCV external device-pointer binding is missing")
    if not hasattr(cv2.cuda, "nms"):
        raise RuntimeError("OpenCV GPU NMS binding is missing")
    if "align_corners" not in (cv2.cuda.resize.__doc__ or ""):
        raise RuntimeError("OpenCV HIP resize align_corners support is missing")

    reader = RocDecodeReader(str(workspace / "data/sidewalk.mp4"), device_id=0)
    try:
        ok, frame = reader.read_gpu()
    finally:
        reader.release()
    if not ok or frame is None or not frame.is_cuda:
        raise RuntimeError("rocDecode did not return a GPU frame")

    bridge_source = workspace / "native/hip_vaapi_bridge.cpp"
    bridge_sha256 = hashlib.sha256(bridge_source.read_bytes()).hexdigest()
    expected_bridge_sha256 = os.environ.get(
        "ULTRALYTICS_HIP_VAAPI_BRIDGE_SHA256", ""
    )
    if not expected_bridge_sha256 or expected_bridge_sha256 == "unknown":
        raise RuntimeError("Image does not declare its HIP/VAAPI bridge identity")
    if bridge_sha256 != expected_bridge_sha256:
        raise RuntimeError(
            "HIP/VAAPI bridge source differs from the image: "
            f"source={bridge_sha256}, image={expected_bridge_sha256}"
        )

    direct_output = workspace / "output/.image-direct-encode-smoke.mp4"
    direct_output.parent.mkdir(parents=True, exist_ok=True)
    direct_output.unlink(missing_ok=True)
    encoder = HipVaapiEncoder(
        str(direct_output),
        os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128"),
        int(frame.shape[1]),
        int(frame.shape[0]),
        25.0,
        24,
    )
    try:
        encoder.write(
            frame.data_ptr(),
            int(frame.stride(0)),
            [],
            [],
            torch.cuda.current_stream().cuda_stream,
            ["DIRECT ENCODE SMOKE"],
        )
        direct_encode_info = dict(encoder.info())
    finally:
        encoder.close()
    direct_capture = cv2.VideoCapture(str(direct_output))
    direct_ok, direct_frame = direct_capture.read()
    direct_frame_count = int(direct_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    direct_capture.release()
    direct_output.unlink(missing_ok=True)
    if not direct_ok or direct_frame is None or direct_frame_count != 1:
        raise RuntimeError(
            f"HIP/VAAPI smoke output is invalid: ok={direct_ok}, "
            f"frames={direct_frame_count}"
        )
    if tuple(direct_frame.shape[:2]) != tuple(frame.shape[:2]):
        raise RuntimeError(
            f"HIP/VAAPI smoke shape mismatch: {direct_frame.shape} vs {frame.shape}"
        )

    cache_root = Path(
        os.environ.get(
            "ULTRALYTICS_MIGRAPHX_CACHE_ROOT",
            workspace / "models/ort-migraphx-cache",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    backend = validate_backend(workspace / "models/yolo26x.onnx", iterations=3)

    result = {
        "workspace": str(workspace),
        "ultralytics": ultralytics.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "opencv": cv2.__version__,
        "jupyterlab": jupyterlab.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "decoded_frame": list(frame.shape),
        "hip_vaapi_bridge": {
            "module": str(Path(__import__("hip_vaapi_bridge").__file__).resolve()),
            "sha256": bridge_sha256,
            "decoded_frames": direct_frame_count,
            "encoded_frame": list(direct_frame.shape),
            "encoder": direct_encode_info,
        },
        "migraphx_cache_root": str(cache_root),
        "backend": backend,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
