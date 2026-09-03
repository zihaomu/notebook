#!/usr/bin/env python3
"""Validate the H.264 output produced from HIP-mapped VAAPI surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Cannot decode the first probe frame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    margin = 64
    quarter = width // 4
    means = [
        float(np.mean(gray[margin : height - margin, index * quarter + margin : (index + 1) * quarter - margin]))
        for index in range(4)
    ]
    if width != 1920 or height != 1080:
        raise RuntimeError(f"Unexpected dimensions: {width}x{height}")
    if frame_count != args.frames:
        raise RuntimeError(f"Unexpected frame count: {frame_count} != {args.frames}")
    if not all(right - left > 25 for left, right in zip(means, means[1:])):
        raise RuntimeError(f"Encoded luma bars are invalid: {means}")

    print(json.dumps({
        "video": str(args.video),
        "width": width,
        "height": height,
        "frames": frame_count,
        "quarter_luma_means": [round(value, 2) for value in means],
    }, indent=2))
    print("VAAPI_HIP_ENCODE_PIXELS=PASS")


if __name__ == "__main__":
    main()
