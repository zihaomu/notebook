"""Shared path and runtime discovery for the OpenCV AMD GPU notebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_package_root() -> Path:
    configured = os.environ.get("OPENCV_AMD_END2END_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _first_existing(*candidates: str | Path) -> Path:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return Path(candidates[0]).expanduser().resolve()


ROOT = _find_package_root()
SRC = ROOT / "src"
DATA = ROOT / "data"
MODELS = Path(os.environ.get("OPENCV_AMD_END2END_MODEL_DIR", ROOT / "models")).resolve()
OUTPUT = Path(os.environ.get("OPENCV_AMD_END2END_OUTPUT_DIR", ROOT / "output")).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)

OPENCV_INSTALL = _first_existing(
    os.environ.get("OPENCV_INSTALL", "/opt/opencv5"),
    "/opencv_workspace/install",
)
OPENCV_INCLUDE = OPENCV_INSTALL / "include/opencv5"
OPENCV_LIB = OPENCV_INSTALL / "lib"

python_paths = [
    OPENCV_LIB / "python3.10/site-packages",
    OPENCV_LIB / "python3.12/site-packages",
    OPENCV_LIB / "python3.12/dist-packages",
    Path("/opt/rocm/lib"),
    SRC,
]
existing_python_paths = [str(path) for path in python_paths if path.exists()]
for value in existing_python_paths:
    if value not in sys.path:
        sys.path.insert(0, value)
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = ":".join(
    [*existing_python_paths, *([existing_pythonpath] if existing_pythonpath else [])]
)

library_paths = [str(OPENCV_LIB), "/opt/rocm/lib"]
existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [*library_paths, *([existing_ld] if existing_ld else [])]
)
os.environ.setdefault("OPENCV_AMD_END2END_ROOT", str(ROOT))
os.environ.setdefault("OPENCV_AMD_END2END_MODEL_DIR", str(MODELS))
os.environ.setdefault("OPENCV_AMD_END2END_OUTPUT_DIR", str(OUTPUT))
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SOURCE_VIDEO = DATA / "sidewalk.mp4"
YOLO_ONNX = MODELS / "yolo26x.onnx"
YOLO_COMPILED = MODELS / "yolo26x_compiled.mxr"
QWEN_GGUF = MODELS / "Qwen3-VL-8B-Instruct-Q8_0.gguf"
QWEN_MMPROJ = MODELS / "mmproj-F16.gguf"
LLAMACPP_ROOT_URL = os.environ.get("LLAMACPP_ROOT_URL", "http://zihao_llamacpp_q8:8199")
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", f"{LLAMACPP_ROOT_URL}/v1")


def require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required files:\n  - {joined}")


def package_layout() -> dict[str, str]:
    return {
        "root": str(ROOT),
        "src": str(SRC),
        "data": str(DATA),
        "models": str(MODELS),
        "output": str(OUTPUT),
        "opencv_install": str(OPENCV_INSTALL),
        "llamacpp": LLAMACPP_BASE_URL,
    }
