"""Reusable orchestration helpers for the pipeline notebooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts import notebook_env as env


RUN_DIR = env.OUTPUT / "pipeline"
YOLO_VIDEO = RUN_DIR / "01_yolo_base.mp4"
PIPELINE_LOG = RUN_DIR / "01_yolo_base.log"
TIMELINE = RUN_DIR / "02_llamacpp_q8_timeline.json"
SUBTITLE_LOG = RUN_DIR / "02_llamacpp_q8_subtitle.log"
FINAL_VIDEO = RUN_DIR / "03_final_llamacpp_q8.mp4"


def video_info(path: Path) -> dict[str, float | int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fps),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_seconds"] = info["frames"] / info["fps"]
    capture.release()
    return info


def pipeline_command(
    source: Path = env.SOURCE_VIDEO,
    output: Path = YOLO_VIDEO,
    max_frames: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        str(env.SRC / "pipeline.py"),
        "--input", str(source),
        "--output", str(output),
        "--no-vlm",
        "--video-decode", os.environ.get("VIDEO_DECODE", "rocdecode"),
        "--video-encode", os.environ.get("VIDEO_ENCODE", "vaapi"),
    ]
    if max_frames:
        command.extend(("--max-frames", str(max_frames)))
    return command


def subtitle_command(
    source: Path = env.SOURCE_VIDEO,
    annotated_input: Path = YOLO_VIDEO,
    output: Path = FINAL_VIDEO,
    force_analyze: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(env.SRC / "scene_subtitle_pipeline.py"),
        "--mode", "both",
        "--source", str(source),
        "--annotated-input", str(annotated_input),
        "--output", str(output),
        "--timeline", str(TIMELINE),
        "--artifact-dir", str(RUN_DIR),
        "--interval", os.environ.get("SCENE_INTERVAL", "4"),
        "--video-encode", os.environ.get("VIDEO_ENCODE", "vaapi"),
        "--vlm-url", env.LLAMACPP_BASE_URL,
    ]
    if force_analyze:
        command.append("--force-analyze")
    return command


def run_and_log(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=env.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return_code = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return time.perf_counter() - started


def required_outputs() -> tuple[Path, ...]:
    return YOLO_VIDEO, PIPELINE_LOG, TIMELINE, SUBTITLE_LOG, FINAL_VIDEO


def run_workflow(force: bool = False) -> dict[str, object]:
    env.require_files(
        env.SOURCE_VIDEO,
        env.YOLO_ONNX,
        env.YOLO_COMPILED,
        env.QWEN_GGUF,
        env.QWEN_MMPROJ,
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    complete = all(path.is_file() for path in required_outputs())
    pipeline_seconds = None
    subtitle_seconds = None
    if force or not complete:
        pipeline_seconds = run_and_log(pipeline_command(), PIPELINE_LOG)
        subtitle_seconds = run_and_log(
            subtitle_command(force_analyze=True), SUBTITLE_LOG
        )
    result = validate_outputs()
    result.update({
        "reused": complete and not force,
        "pipeline_seconds": pipeline_seconds,
        "subtitle_seconds": subtitle_seconds,
    })
    return result


def validate_outputs() -> dict[str, object]:
    missing = [str(path) for path in required_outputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing workflow outputs:\n  - " + "\n  - ".join(missing)
        )
    log = PIPELINE_LOG.read_text(encoding="utf-8", errors="replace")
    timeline = json.loads(TIMELINE.read_text(encoding="utf-8"))
    source_info = video_info(env.SOURCE_VIDEO)
    yolo_info = video_info(YOLO_VIDEO)
    final_info = video_info(FINAL_VIDEO)
    if "decode: rocdecode" not in log or "encode: vaapi" not in log:
        raise RuntimeError("The saved run did not use rocDecode + VA-API")
    if timeline.get("backend") != "llamacpp":
        raise RuntimeError(f"Unexpected VLM backend: {timeline.get('backend')}")
    if "Q8_0.gguf" not in timeline.get("model", ""):
        raise RuntimeError(f"Unexpected VLM model: {timeline.get('model')}")
    if source_info["frames"] != yolo_info["frames"] or source_info["frames"] != final_info["frames"]:
        raise RuntimeError("Input/output frame counts differ")
    return {
        "source": source_info,
        "yolo": yolo_info,
        "final": final_info,
        "segments": len(timeline.get("segments", [])),
        "backend": timeline["backend"],
        "model": timeline["model"],
        "run_dir": str(RUN_DIR),
    }
