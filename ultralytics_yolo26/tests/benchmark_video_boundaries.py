#!/usr/bin/env python3
"""Measure where the production video pipeline crosses the GPU boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from detector import UltralyticsYOLODetector
from postprocess import draw_detections, draw_stats
from preprocess import GPUPreprocessor
from video_io import make_reader, make_writer


def benchmark(mode: str, video: Path, output: Path | None = None) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    reader, decode_kind = make_reader(str(video), 0, prefer_gpu=True)
    writer = None
    encode_kind = None
    if mode == "encode":
        if output is None:
            raise ValueError("--output is required for encode mode")
        writer, encode_kind = make_writer(
            str(output), width, height, fps, prefer_gpu=True,
            device=os.environ.get("VAAPI_DEVICE"),
        )
    processor = GPUPreprocessor((640, 640), "cuda:0")
    detector = UltralyticsYOLODetector(str(PACKAGE_ROOT / "models/yolo26x.onnx"), 0)

    warmup = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device="cuda")
    for _ in range(20):
        detector.infer_gpu(warmup)
    torch.cuda.synchronize()

    frame_count = 0
    started = time.perf_counter()
    try:
        while True:
            ok, rgb_gpu = reader.read_gpu()
            if not ok:
                break
            blob, scale, pad_w, pad_h = processor.process(rgb_gpu)
            raw = detector.infer_gpu(blob)
            detections = detector._parse_gpu(
                raw, scale, pad_w, pad_h, tuple(rgb_gpu.shape)
            )
            if mode != "gpu":
                frame = cv2.cvtColor(rgb_gpu.cpu().numpy(), cv2.COLOR_RGB2BGR)
                draw_detections(frame, detections, names=detector.names)
                draw_stats(
                    frame,
                    {"Mode": mode, "Frame": frame_count + 1, "Detections": len(detections)},
                )
                if writer is not None:
                    writer.write(frame)
            frame_count += 1
    finally:
        reader.release()
    loop_finished = time.perf_counter()
    if writer is not None:
        writer.release()
    finished = time.perf_counter()

    if frame_count != expected_frames:
        raise RuntimeError(f"Processed {frame_count} frames, expected {expected_frames}")
    loop_seconds = loop_finished - started
    total_seconds = finished - started
    return {
        "mode": mode,
        "frames": frame_count,
        "decode": decode_kind,
        "encode": encode_kind,
        "host": {
            "logical_cpus": os.cpu_count(),
            "load_average": [round(value, 2) for value in os.getloadavg()],
        },
        "loop_seconds": round(loop_seconds, 3),
        "loop_fps": round(frame_count / loop_seconds, 2),
        "writer_drain_seconds": round(finished - loop_finished, 3),
        "end_to_end_seconds": round(total_seconds, 3),
        "end_to_end_fps": round(frame_count / total_seconds, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gpu", "overlay", "encode"), required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = benchmark(args.mode, args.video, args.output)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
