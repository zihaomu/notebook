#!/usr/bin/env python3
"""Emit an NVTX range for rocprof memory-copy tracing of steady YOLO inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from detector import UltralyticsYOLODetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    detector = UltralyticsYOLODetector(str(args.model), 0)
    input_tensor = torch.rand((1, 3, 640, 640), dtype=torch.float32, device="cuda")
    for _ in range(args.warmup):
        detector.infer_gpu(input_tensor)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("MEASURE_YOLO")
    for _ in range(args.iterations):
        detector.infer_gpu(input_tensor)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print({"iterations": args.iterations, "input_pointer": input_tensor.data_ptr()})


if __name__ == "__main__":
    main()
