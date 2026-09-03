#!/usr/bin/env python3
"""Benchmark the GPU-resident YOLO video stages without overlay or encoding."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from detector import UltralyticsYOLODetector
from preprocess import GPUPreprocessor
from video_io import RocDecodeReader


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
    }


def benchmark(video: Path, frames: int, warmup: int) -> dict[str, object]:
    reader = RocDecodeReader(str(video), device_id=0)
    processor = GPUPreprocessor((640, 640), device="cuda:0")
    detector = UltralyticsYOLODetector(device_id=0)
    pointers = {
        **processor.pointer_info(),
        "output_pointer": detector.provider_info()["output_pointer"],
    }
    samples = {"decode": [], "preprocess": [], "inference": [], "gpu_nms": []}
    try:
        for index in range(frames + warmup):
            started = time.perf_counter()
            ok, frame = reader.read_gpu()
            if not ok:
                raise RuntimeError(f"Video ended after {index} frames")
            torch.cuda.synchronize()
            decoded = time.perf_counter()

            blob, scale, pad_w, pad_h = processor.process(frame)
            torch.cuda.synchronize()
            preprocessed = time.perf_counter()

            output = detector.infer_gpu(blob)
            torch.cuda.synchronize()
            inferred = time.perf_counter()

            detector._parse_gpu(output, scale, pad_w, pad_h, frame.shape)
            torch.cuda.synchronize()
            finished = time.perf_counter()

            current_pointers = {
                **processor.pointer_info(),
                "output_pointer": output.data_ptr(),
            }
            if current_pointers != pointers:
                raise RuntimeError(
                    f"GPU allocation changed: {current_pointers} != {pointers}"
                )
            if index >= warmup:
                samples["decode"].append((decoded - started) * 1000)
                samples["preprocess"].append((preprocessed - decoded) * 1000)
                samples["inference"].append((inferred - preprocessed) * 1000)
                samples["gpu_nms"].append((finished - inferred) * 1000)
    finally:
        reader.release()

    stage_results = {name: summarize(values) for name, values in samples.items()}
    gpu_path_ms = sum(stage_results[name]["mean_ms"] for name in stage_results)
    return {
        "frames": frames,
        "warmup_frames": warmup,
        "host": {
            "logical_cpus": os.cpu_count(),
            "load_average": [round(value, 2) for value in os.getloadavg()],
        },
        "provider": detector.provider_info(),
        "stable_pointers": pointers,
        "stages": stage_results,
        "gpu_path_mean_ms": round(gpu_path_ms, 3),
        "gpu_path_fps": round(1000 / gpu_path_ms, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = benchmark(args.video, args.frames, args.warmup)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
