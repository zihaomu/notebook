"""Configuration for the llama.cpp Q8_0 Vision AI Pipeline."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(os.environ.get(
    "OPENCV_AMD_END2END_ROOT", Path(__file__).resolve().parents[1]
)).resolve()
MODEL_DIR = Path(os.environ.get(
    "OPENCV_AMD_END2END_MODEL_DIR", PROJECT_ROOT / "models"
)).resolve()
YOLO_MODEL_PATH = str(MODEL_DIR / "yolo26x.onnx")
YOLO_COMPILED_PATH = str(MODEL_DIR / "yolo26x_compiled.mxr")
LLAMACPP_MODEL_PATH = MODEL_DIR / "Qwen3-VL-8B-Instruct-Q8_0.gguf"
LLAMACPP_MMPROJ_PATH = MODEL_DIR / "mmproj-F16.gguf"
QWEN_MODEL_PATH = str(LLAMACPP_MODEL_PATH)
QWEN_MMPROJ_PATH = str(LLAMACPP_MMPROJ_PATH)
EXPECTED_MODEL_BYTES = 8_709_520_224
EXPECTED_MMPROJ_BYTES = 1_159_030_336
MODEL_SHA256 = "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"
MMPROJ_SHA256 = "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"


def validate_model_files():
    """Validate the fixed Q8_0 model and F16 vision projector."""
    expected = {
        LLAMACPP_MODEL_PATH: EXPECTED_MODEL_BYTES,
        LLAMACPP_MMPROJ_PATH: EXPECTED_MMPROJ_BYTES,
    }
    for path, size in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
        if path.stat().st_size != size:
            raise RuntimeError(f"Unexpected size for {path}: {path.stat().st_size} != {size}")
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise RuntimeError(f"Invalid GGUF header: {path}")
    return expected

# --- YOLO Preprocessing ---
INPUT_SIZE = (640, 640)  # (width, height)
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# --- GPU ---
GPU_DEVICE_ID = 0

# --- llama.cpp / Qwen3-VL Q8_0 ---
VLM_BACKEND = "llamacpp"
LLAMACPP_BASE_URL = os.environ.get(
    "LLAMACPP_BASE_URL", "http://zihao_llamacpp_q8:8199/v1"
)
LLAMACPP_MODEL_NAME = "auto"
LLAMACPP_CONTAINER = os.environ.get("LLAMACPP_CONTAINER", "zihao_llamacpp_q8")
LLAMACPP_QUANTIZATION = "Q8_0"

# --- Shared VLM settings ---
VLM_MAX_TOKENS = 100
VLM_TOP_K_ROIS = 3  # max ROIs to send to VLM per frame
VLM_PROMPT = "Describe this object or scene in one concise sentence."

# --- COCO class names (80 classes) ---
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
