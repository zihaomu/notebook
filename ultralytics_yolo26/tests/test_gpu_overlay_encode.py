#!/usr/bin/env python3
"""Run a short GPU overlay plus direct VAAPI encode integration test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "native/build"))

import config
from detector import UltralyticsYOLODetector
from preprocess import GPUPreprocessor
from video_io import RocDecodeReader, make_gpu_writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render-node",
        default=os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128"),
    )
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    reader = RocDecodeReader(str(args.video), device_id=0)
    processor = GPUPreprocessor((640, 640), device="cuda:0")
    detector = UltralyticsYOLODetector(device_id=0)
    encoder, encoder_kind = make_gpu_writer(
        str(args.output), width, height, fps, device=args.render_node
    )
    if encoder_kind != "vaapi-direct":
        raise RuntimeError(f"Unexpected encoder: {encoder_kind}")
    frame_count = 0
    total_detections = 0
    started = time.perf_counter()
    try:
        while frame_count < args.frames:
            ok, rgb_gpu = reader.read_gpu()
            if not ok:
                break
            blob, scale, pad_w, pad_h = processor.process(rgb_gpu)
            output_gpu = detector.infer_gpu(blob)
            detections = detector._parse_gpu(
                output_gpu, scale, pad_w, pad_h, tuple(rgb_gpu.shape)
            )
            encoder.write_gpu(
                rgb_gpu,
                detections,
                detector.names,
                [f"FRAME {frame_count + 1}/{args.frames}"],
            )
            total_detections += len(detections)
            frame_count += 1
    finally:
        reader.release()
        encoder.release()
    elapsed = time.perf_counter() - started
    encoder_info = encoder.info()

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames,nb_read_frames,nb_read_packets",
            "-of", "json", str(args.output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    output_frames = int(stream["nb_read_frames"])
    output_packets = int(stream["nb_read_packets"])
    output_samples = int(stream["nb_frames"])
    output_width = int(stream["width"])
    output_height = int(stream["height"])
    output_capture = cv2.VideoCapture(str(args.output))
    ok, encoded = output_capture.read()
    output_capture.release()
    source_capture = cv2.VideoCapture(str(args.video))
    source_ok, source = source_capture.read()
    source_capture.release()
    if not ok or not source_ok:
        raise RuntimeError("Cannot decode source or direct-encode output")
    absolute_difference = np.abs(
        encoded.astype(np.int16) - source.astype(np.int16)
    )
    mean_absolute_difference = float(np.mean(absolute_difference))
    changed = np.any(absolute_difference >= 24, axis=2)
    channel_spread = (
        encoded.max(axis=2).astype(np.int16)
        - encoded.min(axis=2).astype(np.int16)
    )
    changed_high_chroma_pixels = int(
        np.logical_and(changed, channel_spread >= 96).sum()
    )
    status_near_white = np.all(encoded[:125, :430] >= 220, axis=2)
    status_changed = changed[:125, :430]
    status_changed_near_white_pixels = int(
        np.logical_and(status_changed, status_near_white).sum()
    )
    if (output_frames, output_packets, output_samples) != (frame_count,) * 3:
        raise RuntimeError(
            "Output frame accounting differs: "
            f"decoded={output_frames}, packets={output_packets}, "
            f"samples={output_samples}, expected={frame_count}"
        )
    if (
        encoder_info["submitted_frames"] != frame_count
        or encoder_info["encoded_frames"] != frame_count
    ):
        raise RuntimeError(f"Direct writer queue did not drain: {encoder_info}")
    if (output_width, output_height) != (width, height):
        raise RuntimeError(
            f"Output is {output_width}x{output_height}, expected {width}x{height}"
        )
    if (
        total_detections <= 0
        or mean_absolute_difference <= 0.5
        or changed_high_chroma_pixels <= 1000
        or status_changed_near_white_pixels <= 100
    ):
        raise RuntimeError(
            "Overlay validation failed: "
            f"detections={total_detections}, MAE={mean_absolute_difference}, "
            f"changed_high_chroma={changed_high_chroma_pixels}, "
            f"status_changed_near_white={status_changed_near_white_pixels}"
        )

    print(json.dumps({
        "frames": frame_count,
        "fps": round(frame_count / elapsed, 2),
        "elapsed_seconds": round(elapsed, 3),
        "total_detections": total_detections,
        "encoded_resolution": [output_width, output_height],
        "decoded_frames": output_frames,
        "encoded_packets": output_packets,
        "container_samples": output_samples,
        "first_frame_source_output_mae": round(mean_absolute_difference, 3),
        "changed_high_chroma_pixels": changed_high_chroma_pixels,
        "status_changed_near_white_pixels": status_changed_near_white_pixels,
        "encoder": encoder_info,
    }, indent=2))
    print("GPU_OVERLAY_DIRECT_VAAPI=PASS")


if __name__ == "__main__":
    main()
