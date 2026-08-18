#!/opt/venv/bin/python3
"""Smoke-test the image with the notebook package mounted at /workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    workspace = Path(os.environ.get("OPENCV_AMD_END2END_ROOT", "/workspace")).resolve()
    expected_workspace = Path("/workspace")
    if workspace != expected_workspace:
        raise RuntimeError(f"Expected workspace {expected_workspace}, got {workspace}")

    required_paths = [
        workspace / "scripts/notebook_env.py",
        workspace / "src/video_io.py",
        workspace / "data/sidewalk.mp4",
        workspace / "models/yolo26x.onnx",
        workspace / "models/yolo26x_compiled.mxr",
        Path("/opencv_workspace/third_party/opencv/CMakeLists.txt"),
        Path("/opencv_workspace/third_party/opencv_contrib/modules"),
        Path("/opencv_workspace/THIRD_PARTY_VERSIONS"),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image/package paths: {missing}")

    import cv2
    import jupyterlab
    import migraphx
    import pyRocVideoDecode
    import torch

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

    from video_io import RocDecodeReader

    reader = RocDecodeReader(str(workspace / "data/sidewalk.mp4"), device_id=0)
    try:
        ok, frame = reader.read_gpu()
    finally:
        reader.release()
    if not ok or frame is None or not frame.is_cuda:
        raise RuntimeError("rocDecode did not return a GPU frame")

    versions = {}
    for line in Path("/opencv_workspace/THIRD_PARTY_VERSIONS").read_text().splitlines():
        key, value = line.split("=", 1)
        versions[key] = value

    result = {
        "workspace": str(workspace),
        "opencv": cv2.__version__,
        "jupyterlab": jupyterlab.__version__,
        "opencv_devices": cv2.cuda.getCudaEnabledDeviceCount(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "migraphx": migraphx.__file__,
        "rocdecode": str(Path(pyRocVideoDecode.__path__[0]).resolve()),
        "decoded_frame": list(frame.shape),
        "third_party": versions,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
