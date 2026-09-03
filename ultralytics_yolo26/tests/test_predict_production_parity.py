#!/usr/bin/env python3
"""Compare the familiar YOLO.predict path with the production backend path."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import config
from migraphx_cache import prepare_cache
from detector import UltralyticsYOLODetector
from preprocess import preprocess_frame_cpu


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return float(intersection / union)


def validate(model_path: Path, video_path: Path, minimum_iou: float = 0.90):
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read the first frame of {video_path}")

    prepare_cache(model_path, model_path.parent / "ort-migraphx-cache", 0)
    model = YOLO(str(model_path), task="detect")
    reference_result = model.predict(
        frame,
        device=0,
        half=True,
        imgsz=config.INPUT_SIZE[0],
        conf=config.CONF_THRESHOLD,
        iou=config.NMS_IOU_THRESHOLD,
        verbose=False,
    )[0]
    reference = reference_result.boxes.data.detach().cpu().numpy()
    del model, reference_result
    gc.collect()
    torch.cuda.empty_cache()

    detector = UltralyticsYOLODetector(model_path=str(model_path), device_id=0)
    blob, scale, pad_w, pad_h = preprocess_frame_cpu(frame)
    production = np.asarray(
        detector.detect_and_parse(blob, scale, pad_w, pad_h, frame.shape),
        dtype=np.float32,
    )

    if len(reference) != len(production):
        raise RuntimeError(
            f"Box count differs: predict={len(reference)}, production={len(production)}"
        )
    reference_classes = sorted(reference[:, 5].astype(int).tolist())
    production_classes = sorted(production[:, 5].astype(int).tolist())
    if reference_classes != production_classes:
        raise RuntimeError(
            f"Classes differ: predict={reference_classes}, production={production_classes}"
        )

    best_ious = []
    for reference_box in reference:
        candidates = [
            box_iou(reference_box, production_box)
            for production_box in production
            if int(production_box[5]) == int(reference_box[5])
        ]
        best_ious.append(max(candidates))
    if min(best_ious) < minimum_iou:
        raise RuntimeError(f"Minimum class-matched IoU is too low: {best_ious}")

    return {
        "predict_boxes": len(reference),
        "production_boxes": len(production),
        "classes": reference_classes,
        "minimum_class_matched_iou": round(min(best_ious), 4),
        "mean_class_matched_iou": round(float(np.mean(best_ious)), 4),
        "minimum_required_iou": minimum_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.90)
    args = parser.parse_args()
    print(json.dumps(validate(args.model, args.video, args.minimum_iou), indent=2))


if __name__ == "__main__":
    main()
