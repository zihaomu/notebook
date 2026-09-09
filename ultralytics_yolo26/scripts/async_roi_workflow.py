"""Reusable YOLO-only and asynchronous ROI-VLM workshop experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts import notebook_env as env


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(path: Path) -> dict[str, object]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets", "-count_frames",
            "-select_streams", "v:0", "-show_entries",
            (
                "stream=width,height,codec_name,profile,pix_fmt,r_frame_rate,"
                "duration,nb_frames,nb_read_packets,nb_read_frames"
            ),
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "profile": stream.get("profile"),
        "pixel_format": stream["pix_fmt"],
        "frame_rate": stream["r_frame_rate"],
        "duration_seconds": float(stream["duration"]),
        "container_samples": int(stream["nb_frames"]),
        "packets": int(stream["nb_read_packets"]),
        "decoded_frames": int(stream["nb_read_frames"]),
    }


def experiment_paths(output_dir: Path, mode: str) -> dict[str, Path]:
    stem = mode.replace("-", "_")
    return {
        "video": output_dir / f"{stem}.mp4",
        "metrics": output_dir / f"{stem}_metrics.json",
        "log": output_dir / f"{stem}.log",
    }


def pipeline_command(
    *, output_dir: Path, mode: str, max_frames: int, interval: int = 30,
    top_k: int = 3, drain_timeout: float = 15.0, source: Path | None = None,
) -> tuple[list[str], dict[str, Path]]:
    if mode not in {"yolo-only", "async-roi"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if max_frames < 0:
        raise ValueError("max_frames cannot be negative")
    if interval < 1 or top_k < 1:
        raise ValueError("interval and top_k must be at least 1")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = experiment_paths(output_dir, mode)
    command = [
        sys.executable, str(env.SRC / "pipeline.py"),
        "--input", str(source or env.SOURCE_VIDEO),
        "--output", str(paths["video"]),
        "--video-decode", "rocdecode", "--video-encode", "vaapi",
        "--gpu-direct-encode", "off", "--metrics-json", str(paths["metrics"]),
    ]
    if max_frames:
        command.extend(("--max-frames", str(max_frames)))
    if mode == "yolo-only":
        command.append("--no-vlm")
    else:
        command.extend((
            "--vlm-url", env.LLAMACPP_BASE_URL,
            "--vlm-interval", str(interval),
            "--vlm-top-k", str(top_k),
            "--vlm-drain-timeout", str(drain_timeout),
        ))
    return command, paths


def _expected_frames(source: Path, max_frames: int) -> int:
    import cv2
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return min(total, max_frames) if max_frames else total


def validate_experiment(
    *, paths: dict[str, Path], mode: str, max_frames: int,
    interval: int = 30, top_k: int = 3, source: Path | None = None,
) -> dict[str, object]:
    source = source or env.SOURCE_VIDEO
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {name} artifact: {path}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    expected_frames = _expected_frames(source, max_frames)
    if metrics["frames"] != expected_frames:
        raise RuntimeError(f"Metrics contain {metrics['frames']} frames, expected {expected_frames}")
    video = video_info(paths["video"])
    for key in ("container_samples", "packets", "decoded_frames"):
        if video[key] != expected_frames:
            raise RuntimeError(f"{key}={video[key]}, expected {expected_frames}")
    if mode == "async-roi":
        if metrics.get("vlm_interval_frames") != interval:
            raise RuntimeError(
                f"VLM interval is {metrics.get('vlm_interval_frames')}, expected {interval}"
            )
        if metrics.get("vlm_top_k") != top_k:
            raise RuntimeError(
                f"VLM top-K is {metrics.get('vlm_top_k')}, expected {top_k}"
            )
        vlm = metrics.get("vlm") or {}
        if not metrics.get("vlm_drained"):
            raise RuntimeError("VLM batch did not drain before the deadline")
        if vlm.get("submitted_batches", 0) < 1:
            raise RuntimeError("No VLM batch was submitted")
        if vlm.get("completed_batches") != vlm.get("submitted_batches"):
            raise RuntimeError(f"Incomplete VLM batches: {vlm}")
        if vlm.get("failed_rois") != 0:
            raise RuntimeError(f"VLM ROI failures: {vlm}")
    return {"mode": mode, "metrics": metrics, "video": video}


def run_experiment(
    *, output_dir: Path, mode: str, max_frames: int, interval: int = 30,
    top_k: int = 3, drain_timeout: float = 15.0, force: bool = False,
    source: Path | None = None,
) -> dict[str, object]:
    command, paths = pipeline_command(
        output_dir=output_dir, mode=mode, max_frames=max_frames,
        interval=interval, top_k=top_k, drain_timeout=drain_timeout,
        source=source,
    )
    reuse = False
    if not force and all(path.is_file() for path in paths.values()):
        try:
            validate_experiment(
                paths=paths,
                mode=mode,
                max_frames=max_frames,
                interval=interval,
                top_k=top_k,
                source=source,
            )
            reuse = True
        except (KeyError, TypeError, ValueError, OSError, RuntimeError, json.JSONDecodeError):
            reuse = False
    if not reuse:
        started = time.perf_counter()
        process = subprocess.run(
            command, cwd=env.ROOT, env=os.environ.copy(),
            capture_output=True, text=True,
        )
        paths["log"].write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"{mode} failed with exit code {process.returncode}; see {paths['log']}"
            )
        elapsed = time.perf_counter() - started
    else:
        elapsed = None
    result = validate_experiment(
        paths=paths, mode=mode, max_frames=max_frames,
        interval=interval, top_k=top_k, source=source,
    )
    result.update({
        "command": command,
        "paths": {name: str(path) for name, path in paths.items()},
        "execution_seconds": None if elapsed is None else round(elapsed, 3),
    })
    return result


def comparison(yolo_only: dict[str, object], async_roi: dict[str, object]) -> dict[str, object]:
    baseline = yolo_only["metrics"]
    parallel = async_roi["metrics"]
    return {
        "frames": parallel["frames"],
        "yolo_only_fps": baseline["fps"],
        "parallel_fps": parallel["fps"],
        "fps_retained_percent": round(parallel["fps"] / baseline["fps"] * 100, 1),
        "yolo_only_detection_ms": baseline["detection_mean_ms"],
        "parallel_detection_ms": parallel["detection_mean_ms"],
        "parallel_detection_idle": parallel["detection_vlm_idle"],
        "parallel_detection_active": parallel["detection_vlm_active"],
        "trigger_opportunities": parallel["trigger_opportunities"],
        "trigger_with_detections": parallel["trigger_with_detections"],
        **{f"vlm_{key}": value for key, value in (parallel.get("vlm") or {}).items()},
    }


def write_manifest(
    *, output_dir: Path, results: dict[str, dict[str, object]],
    settings: dict[str, object],
) -> Path:
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "workflow": "async-roi-vlm",
        "settings": settings,
        "results": results,
        "artifacts": artifacts,
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest
