"""Shared paths and runtime discovery for the Ultralytics YOLO26 workshop."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_package_root(path: Path) -> bool:
    return (
        (path / "scripts/notebook_env.py").is_file()
        and (path / "src").is_dir()
        and (path / "data").is_dir()
    )


def _find_package_root() -> tuple[Path, Path | None]:
    configured = os.environ.get("ULTRALYTICS_YOLO26_ROOT") or os.environ.get("OPENCV_AMD_END2END_ROOT")
    configured_path = Path(configured).expanduser().resolve() if configured else None
    if configured_path is not None and _is_package_root(configured_path):
        return configured_path, configured_path

    package_root = Path(__file__).resolve().parents[1]
    if not _is_package_root(package_root):
        raise FileNotFoundError(f"Invalid ultralytics_yolo26 package root: {package_root}")
    return package_root, configured_path


def _project_path(environment_name: str, directory_name: str) -> Path:
    configured = os.environ.get(environment_name)
    default = ROOT / directory_name
    if not configured:
        return default.resolve()

    configured_path = Path(configured).expanduser().resolve()
    if STALE_ROOT is not None and not _is_package_root(STALE_ROOT):
        if configured_path == (STALE_ROOT / directory_name).resolve():
            return default.resolve()
    return configured_path


def _first_existing(*candidates: str | Path) -> Path:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return Path(candidates[0]).expanduser().resolve()


ROOT, STALE_ROOT = _find_package_root()
SRC = ROOT / "src"
DATA = ROOT / "data"
MODELS = _project_path("ULTRALYTICS_YOLO26_MODEL_DIR", "models")
OUTPUT = _project_path("ULTRALYTICS_YOLO26_OUTPUT_DIR", "output")
MODELS.mkdir(parents=True, exist_ok=True)
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
os.environ["ULTRALYTICS_YOLO26_ROOT"] = str(ROOT)
os.environ["ULTRALYTICS_YOLO26_MODEL_DIR"] = str(MODELS)
os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(OUTPUT)
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SOURCE_VIDEO = DATA / "sidewalk.mp4"
YOLO_CHECKPOINT = MODELS / "yolo26x.pt"
YOLO_ONNX = MODELS / "yolo26x.onnx"
MIGRAPHX_CACHE = MODELS / "ort-migraphx-cache"
QWEN_GGUF = MODELS / "Qwen3-VL-8B-Instruct-Q8_0.gguf"
QWEN_MMPROJ = MODELS / "mmproj-F16.gguf"
LLAMACPP_ROOT_URL = os.environ.get("LLAMACPP_ROOT_URL", "http://127.0.0.1:8199")
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
