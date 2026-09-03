#!/opt/venv/bin/python3
"""Smoke-test the Ultralytics YOLO26x Radeon workshop image."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    workspace = Path(
        os.environ.get("ULTRALYTICS_YOLO26_ROOT", "/workspace")
    ).resolve()
    if workspace != Path("/workspace"):
        raise RuntimeError(f"Expected workspace /workspace, got {workspace}")
    seed_root = Path("/opt/ultralytics-yolo26/seed")
    if os.environ.get("ULTRALYTICS_WORKSHOP_BUNDLE") != "baked":
        raise RuntimeError("Image does not declare a baked workshop bundle")

    required_paths = [
        workspace / "scripts/notebook_env.py",
        workspace / "tests/test_ultralytics_migraphx_backend.py",
        workspace / "src/video_io.py",
        workspace / "data/sidewalk.mp4",
        workspace / "models/yolo26x.pt",
        workspace / "models/yolo26x.onnx",
        workspace / "models/ort-migraphx-cache/735f1583e99dfeb733da/identity.json",
        workspace / "models/ort-migraphx-cache/735f1583e99dfeb733da/20e00-58de11c69ae52cf2-9880cf1608079e0d-36a8840bfe2de0d1.mxr",
        workspace / "ultralytics_yolo26x_step_by_step.ipynb",
        workspace / "ultralytics_yolo26x_end_to_end.ipynb",
        seed_root / "scripts/workshop_bundle_identity.py",
        Path("/usr/local/bin/vaapi-hip-encode-probe"),
        Path("/usr/local/share/ultralytics-yolo26-bundle.sha256"),
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
    from scripts.workshop_bundle_identity import compute as compute_bundle_sha256
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

    identities = {
        "bundle": (
            compute_bundle_sha256(seed_root),
            os.environ.get("ULTRALYTICS_WORKSHOP_BUNDLE_SHA256", ""),
        ),
        "checkpoint": (
            hashlib.sha256((workspace / "models/yolo26x.pt").read_bytes()).hexdigest(),
            os.environ.get("ULTRALYTICS_YOLO26_CHECKPOINT_SHA256", ""),
        ),
        "onnx": (
            hashlib.sha256((workspace / "models/yolo26x.onnx").read_bytes()).hexdigest(),
            os.environ.get("ULTRALYTICS_YOLO26_ONNX_SHA256", ""),
        ),
        "migraphx_cache": (
            hashlib.sha256(
                (workspace / "models/ort-migraphx-cache/735f1583e99dfeb733da/20e00-58de11c69ae52cf2-9880cf1608079e0d-36a8840bfe2de0d1.mxr").read_bytes()
            ).hexdigest(),
            os.environ.get("ULTRALYTICS_MIGRAPHX_CACHE_SHA256", ""),
        ),
    }
    for name, (actual, expected) in identities.items():
        if not expected or expected == "unknown" or actual != expected:
            raise RuntimeError(
                f"{name} identity mismatch: actual={actual}, expected={expected}"
            )
    declared_bundle = Path(
        "/usr/local/share/ultralytics-yolo26-bundle.sha256"
    ).read_text(encoding="utf-8").strip()
    if declared_bundle != identities["bundle"][0]:
        raise RuntimeError("Installed bundle identity file does not match the seed")
    for qwen_name in ("Qwen3-VL-8B-Instruct-Q8_0.gguf", "mmproj-F16.gguf"):
        if (seed_root / "models" / qwen_name).exists():
            raise RuntimeError(f"Qwen runtime model was unexpectedly baked: {qwen_name}")

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

    probe_output = Path("/tmp/vaapi-hip-encode-probe.mp4")
    probe_output.unlink(missing_ok=True)
    probe = subprocess.run(
        [
            "/usr/local/bin/vaapi-hip-encode-probe",
            os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128"),
            str(probe_output),
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe_capture = cv2.VideoCapture(str(probe_output))
    probe_frames = int(probe_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    probe_ok, probe_frame = probe_capture.read()
    probe_capture.release()
    probe_output.unlink(missing_ok=True)
    if not probe_ok or probe_frame is None or probe_frames != 4:
        raise RuntimeError(
            f"Standalone VAAPI/HIP probe is invalid: ok={probe_ok}, frames={probe_frames}"
        )
    gray = cv2.cvtColor(probe_frame, cv2.COLOR_BGR2GRAY)
    margin = 64
    quarter = gray.shape[1] // 4
    luma_means = [
        float(np.mean(gray[margin:-margin, index * quarter + margin:(index + 1) * quarter - margin]))
        for index in range(4)
    ]
    if not all(right - left > 25 for left, right in zip(luma_means, luma_means[1:])):
        raise RuntimeError(f"Standalone VAAPI/HIP probe pixels are invalid: {luma_means}")

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
        "baked_bundle": {
            "seed_root": str(seed_root),
            "identities": {
                name: actual for name, (actual, _) in identities.items()
            },
            "qwen_models": "runtime-volume",
        },
        "standalone_vaapi_hip_probe": {
            "binary": "/usr/local/bin/vaapi-hip-encode-probe",
            "frames": probe_frames,
            "quarter_luma_means": [round(value, 2) for value in luma_means],
            "stdout": probe.stdout.strip(),
        },
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
