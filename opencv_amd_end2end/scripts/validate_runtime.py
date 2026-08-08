#!/usr/bin/env python3
"""Validate the runtime used by the pipeline notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import json
from scripts import notebook_env as env


def validate(require_models: bool = False, require_vlm: bool = False) -> dict[str, object]:
    import cv2
    import migraphx
    import requests
    import torch

    env.require_files(env.SOURCE_VIDEO)
    if require_models:
        env.require_files(
            env.YOLO_ONNX,
            env.YOLO_COMPILED,
            env.QWEN_GGUF,
            env.QWEN_MMPROJ,
        )
    gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
    if gpu_count < 1:
        raise RuntimeError("No HIP device is visible through cv2.cuda")
    result: dict[str, object] = {
        **env.package_layout(),
        "python": sys.executable,
        "opencv": cv2.__version__,
        "opencv_hip_devices": gpu_count,
        "torch": torch.__version__,
        "torch_gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "migraphx": migraphx.__file__,
    }
    if require_vlm:
        response = requests.get(f"{env.LLAMACPP_BASE_URL}/models", timeout=10)
        response.raise_for_status()
        payload = response.json()
        model = payload["data"][0]
        capabilities = payload["models"][0]["capabilities"]
        if model.get("owned_by") != "llamacpp" or "Q8_0.gguf" not in model.get("id", ""):
            raise RuntimeError(f"Unexpected VLM service: {model}")
        if "multimodal" not in capabilities:
            raise RuntimeError(f"VLM is not multimodal: {capabilities}")
        result["vlm_model"] = model["id"]
        result["vlm_capabilities"] = capabilities
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-vlm", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.require_models, args.require_vlm), indent=2))


if __name__ == "__main__":
    main()
