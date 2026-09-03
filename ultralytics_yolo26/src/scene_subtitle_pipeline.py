#!/usr/bin/env python3
"""Two-pass Qwen3-VL scene subtitles for the OpenCV pipeline demo."""

import argparse
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from video_io import make_writer
from vlm_client import LlamaCppVLMClient

DEFAULT_PROMPT = (
    "The three panels are chronological frames from the same video segment. "
    "Write one concise English subtitle that summarizes the main subjects, actions, "
    "and setting, emphasizing what is happening. Do not list every object. "
    "Do not mention images, frames, or the model. Do not invent unclear details. "
    "Use at most 18 words and output only the subtitle text."
)
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate timestamped Qwen3-VL scene subtitles and render them below a video."
    )
    parser.add_argument("--mode", choices=["analyze", "render", "both"], default="both")
    parser.add_argument("--source", required=True, help="Original video analyzed by Qwen3-VL")
    parser.add_argument(
        "--annotated-input",
        help="Video receiving subtitles, normally the YOLO-only pipeline output",
    )
    parser.add_argument("--output", help="Rendered subtitle video")
    parser.add_argument("--timeline", required=True, help="Subtitle timeline JSON")
    parser.add_argument("--artifact-dir", required=True, help="Directory for storyboards and SRT")
    parser.add_argument("--interval", type=float, default=4.0, help="Seconds per subtitle")
    parser.add_argument("--band-height", type=int, default=140)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--vlm-url", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--force-analyze", action="store_true")
    parser.add_argument("--video-encode", choices=["auto", "vaapi", "cpu"], default="auto")
    return parser.parse_args()


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


def clean_caption(text):
    caption = " ".join(text.strip().splitlines()).strip()
    caption = re.sub(r"^(subtitle|scene|description)\s*:\s*", "", caption, flags=re.IGNORECASE)
    caption = caption.strip("\"'“”‘’ ")
    if caption.startswith("[VLM error"):
        raise RuntimeError(caption)
    if not caption:
        raise RuntimeError("VLM returned an empty scene subtitle")
    return caption


def read_frame_at(capture, seconds):
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame at {seconds:.3f}s")
    return frame


def build_storyboard(capture, start, end, fps):
    segment_duration = max(end - start, 1.0 / fps)
    sample_times = [
        min(end - 1.0 / fps, start + segment_duration * fraction)
        for fraction in (0.15, 0.50, 0.85)
    ]
    panels = []
    for index, sample_time in enumerate(sample_times, start=1):
        frame = read_frame_at(capture, max(start, sample_time))
        frame = cv2.resize(frame, (600, 338), interpolation=cv2.INTER_AREA)
        cv2.circle(frame, (38, 38), 24, (25, 25, 25), -1)
        cv2.putText(
            frame,
            str(index),
            (29, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        panels.append(frame)
    return cv2.hconcat(panels), sample_times


def srt_timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path, segments):
    lines = []
    for index, segment in enumerate(segments, start=1):
        lines.extend([
            str(index),
            f"{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}",
            segment["caption"],
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args):
    timeline_path = Path(args.timeline)
    if timeline_path.exists() and not args.force_analyze:
        print(f"[subtitle] Reusing timeline: {timeline_path}")
        return json.loads(timeline_path.read_text(encoding="utf-8"))

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = video_metadata(args.source)
    segment_count = math.ceil(metadata["duration"] / args.interval)
    capture = cv2.VideoCapture(args.source)
    config.validate_model_files()
    client = LlamaCppVLMClient(base_url=args.vlm_url, model_name=args.vlm_model)
    if not client.health_check():
        raise RuntimeError(f"VLM unavailable: {client.base_url}")

    segments = []
    for index in range(segment_count):
        start = index * args.interval
        end = min((index + 1) * args.interval, metadata["duration"])
        storyboard, sample_times = build_storyboard(capture, start, end, metadata["fps"])
        storyboard_path = artifact_dir / f"step-13-vlm-scene-input-segment-{index + 1:02d}.jpg"
        if not cv2.imwrite(
            str(storyboard_path), storyboard, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"Cannot write storyboard: {storyboard_path}")
        started = time.perf_counter()
        raw_response = client.describe_roi(storyboard, args.prompt)
        latency = time.perf_counter() - started
        caption = clean_caption(raw_response)
        segment = {
            "index": index + 1,
            "start": start,
            "end": end,
            "sample_times": sample_times,
            "storyboard": storyboard_path.name,
            "caption": caption,
            "raw_response": raw_response,
            "latency_seconds": latency,
        }
        segments.append(segment)
        print(f"[subtitle] {start:5.2f}-{end:5.2f}s ({latency:.2f}s): {caption}")
    capture.release()

    timeline = {
        "source": str(args.source),
        "backend": "llamacpp",
        "model": client.model_name,
        "base_url": client.base_url,
        "prompt": args.prompt,
        "interval_seconds": args.interval,
        "video": metadata,
        "segments": segments,
    }
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(
        json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_srt(artifact_dir / "step-13-vlm-scene-subtitles.srt", segments)
    print(f"[subtitle] Timeline saved: {timeline_path}")
    return timeline


def fit_font(font_path, text, max_width, initial_size):
    size = initial_size
    while size >= 24:
        font = ImageFont.truetype(font_path, size)
        box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, 24)


def render_band(width, height, segment, font_path):
    image = Image.new("RGB", (width, height), (15, 18, 22))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 3), fill=(38, 208, 142))
    label_font = ImageFont.truetype(font_path, 25)
    caption_font = fit_font(font_path, segment["caption"], width - 96, 42)
    draw.text((48, 17), "QWEN3-VL  SCENE UNDERSTANDING", font=label_font, fill=(94, 224, 166))
    draw.text((48, 58), segment["caption"], font=caption_font, fill=(245, 247, 250))
    time_text = f"{segment['start']:04.1f}s - {segment['end']:04.1f}s"
    time_font = ImageFont.truetype(font_path, 20)
    time_box = draw.textbbox((0, 0), time_text, font=time_font)
    draw.text((width - (time_box[2] - time_box[0]) - 48, 22), time_text,
              font=time_font, fill=(145, 154, 166))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(args, timeline):
    if not args.annotated_input or not args.output:
        raise ValueError("--annotated-input and --output are required for rendering")
    font_path = Path(args.font)
    if not font_path.is_file():
        raise FileNotFoundError(f"Subtitle font not found: {font_path}")

    capture = cv2.VideoCapture(args.annotated_input)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open annotated video: {args.annotated_input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or timeline["video"]["fps"]
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_height = height + args.band_height
    prefer_gpu = args.video_encode in ("auto", "vaapi")
    writer, encoder = make_writer(
        args.output, width, output_height, fps, prefer_gpu=prefer_gpu
    )
    if args.video_encode == "vaapi" and encoder != "vaapi":
        raise RuntimeError("VA-API encoding requested but unavailable")

    segments = timeline["segments"]
    bands = [render_band(width, args.band_height, segment, str(font_path))
             for segment in segments]
    segment_index = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        seconds = frame_index / fps
        while (segment_index + 1 < len(segments)
               and seconds >= segments[segment_index]["end"]):
            segment_index += 1
        canvas = np.vstack((frame, bands[segment_index]))
        writer.write(canvas)
        frame_index += 1
    capture.release()
    writer.release()
    if frame_index != total_frames:
        raise RuntimeError(f"Rendered {frame_index} of {total_frames} frames")
    print(
        f"[subtitle] Rendered {frame_index} frames, {width}x{output_height}, "
        f"encoder={encoder}: {args.output}"
    )


def main():
    args = parse_args()
    if args.mode in ("analyze", "both"):
        timeline = analyze(args)
    else:
        timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    if args.mode in ("render", "both"):
        render(args, timeline)


if __name__ == "__main__":
    main()
