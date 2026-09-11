"""Reusable orchestration and artifact identity for the workshop notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts import notebook_env as env


RUN_DIR = Path(
    os.environ.get("ULTRALYTICS_YOLO26_PIPELINE_DIR", env.OUTPUT / "pipeline")
).resolve()
YOLO_VIDEO = RUN_DIR / "01_yolo_base.mp4"
PIPELINE_LOG = RUN_DIR / "01_yolo_base.log"
TIMELINE = RUN_DIR / "02_llamacpp_q8_timeline.json"
SUBTITLE_LOG = RUN_DIR / "02_llamacpp_q8_subtitle.log"
FINAL_VIDEO = RUN_DIR / "03_final_llamacpp_q8.mp4"
MANIFEST = RUN_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(path: Path) -> dict[str, object]:
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
    capture.release()
    info["duration_seconds"] = round(info["frames"] / info["fps"], 3)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets", "-count_frames",
            "-select_streams", "v:0", "-show_entries",
            (
                "stream=codec_name,profile,pix_fmt,r_frame_rate,duration,"
                "nb_frames,nb_read_packets,nb_read_frames"
            ),
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    try:
        info.update({
            "codec": stream["codec_name"],
            "profile": stream["profile"],
            "pixel_format": stream["pix_fmt"],
            "frame_rate": stream["r_frame_rate"],
            "probe_duration_seconds": float(stream["duration"]),
            "container_samples": int(stream["nb_frames"]),
            "packets": int(stream["nb_read_packets"]),
            "decoded_frames": int(stream["nb_read_frames"]),
        })
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Incomplete ffprobe frame accounting for {path}") from error
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
        "--gpu-direct-encode", "on",
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


def required_outputs(include_manifest: bool = True) -> tuple[Path, ...]:
    outputs = (YOLO_VIDEO, PIPELINE_LOG, TIMELINE, SUBTITLE_LOG, FINAL_VIDEO)
    return (*outputs, MANIFEST) if include_manifest else outputs


def current_identity() -> dict[str, object]:
    import onnxruntime as ort
    import torch
    import ultralytics
    from migraphx_cache import runtime_identity

    identity = runtime_identity(env.YOLO_ONNX, 0)
    bridge_source = env.ROOT / "native" / "hip_vaapi_bridge.cpp"
    vision_sources = (
        env.SRC / "pipeline.py",
        env.SRC / "video_io.py",
        env.SRC / "detector.py",
        env.SRC / "preprocess.py",
        env.SRC / "postprocess.py",
        bridge_source,
    )
    identity.update({
        "checkpoint_sha256": sha256(env.YOLO_CHECKPOINT),
        "hip_vaapi_bridge_sha256": sha256(bridge_source),
        "vision_source_sha256": {
            str(path.relative_to(env.ROOT)): sha256(path)
            for path in vision_sources
        },
        "onnxruntime_providers": ort.get_available_providers(),
        "torch_gpu": torch.cuda.get_device_name(0),
        "ultralytics": ultralytics.__version__,
    })
    return identity


def manifest_matches_current() -> tuple[bool, str]:
    if not MANIFEST.is_file():
        return False, "manifest missing"
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        actual = manifest["runtime_identity"]
        expected = current_identity()
        keys = (
            "onnx_sha256",
            "checkpoint_sha256",
            "gpu_arch",
            "gpu_name",
            "torch_hip",
            "migraphx",
            "onnxruntime",
            "ultralytics",
            "ultralytics_commit",
            "patch_sha256",
            "hip_vaapi_bridge_sha256",
            "vision_source_sha256",
            "provider",
            "fp16",
            "input_shape",
            "output_shape",
        )
        mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
        if mismatches:
            return False, "identity mismatch: " + ", ".join(mismatches)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        return False, f"invalid manifest: {type(error).__name__}: {error}"
    return True, "ready for current model/GPU/runtime"


def _pipeline_performance(log: str) -> dict[str, float | int]:
    patterns = {
        "fps": r"Avg FPS:\s*([0-9.]+)",
        "preprocess_ms": r"Avg Preprocess:\s*([0-9.]+)ms",
        "detection_ms": r"Avg Detection:\s*([0-9.]+)ms",
        "frame_d2h_ms": r"Avg Frame D2H:\s*([0-9.]+)ms",
        "overlay_ms": r"Avg Overlay:\s*([0-9.]+)ms",
        "encode_feed_ms": r"Avg Encode feed:\s*([0-9.]+)ms",
        "direct_encode_feed_ms": r"Avg Direct encode feed:\s*([0-9.]+)ms",
        "gpu_overlay_direct_encode_worker_ms": (
            r"Avg GPU overlay \+ direct encode worker:\s*([0-9.]+)ms"
        ),
        "host_load_1m": r"Host load average:\s*([0-9.]+)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log)
        if match:
            result[key] = float(match.group(1))
    queue_match = re.search(
        r"Direct encode queue: depth=(\d+) \| submitted=(\d+) \| encoded=(\d+)",
        log,
    )
    if queue_match:
        result.update({
            "direct_encode_queue_depth": int(queue_match.group(1)),
            "direct_encode_submitted_frames": int(queue_match.group(2)),
            "direct_encode_encoded_frames": int(queue_match.group(3)),
        })
    return result


def write_manifest() -> dict[str, object]:
    timeline = json.loads(TIMELINE.read_text(encoding="utf-8"))
    log = PIPELINE_LOG.read_text(encoding="utf-8", errors="replace")
    identity = current_identity()
    manifest = {
        "schema_version": 3,
        "status": "PASS",
        "package": "ultralytics_yolo26",
        "runtime_identity": identity,
        "models": {
            "yolo26x.pt": {
                "bytes": env.YOLO_CHECKPOINT.stat().st_size,
                "sha256": identity["checkpoint_sha256"],
            },
            "yolo26x.onnx": {
                "bytes": env.YOLO_ONNX.stat().st_size,
                "sha256": identity["onnx_sha256"],
            },
        },
        "pipeline": {
            "source": video_info(env.SOURCE_VIDEO),
            "yolo_video": video_info(YOLO_VIDEO),
            "final_video": video_info(FINAL_VIDEO),
            "performance": _pipeline_performance(log),
            "segments": len(timeline.get("segments", [])),
            "backend": timeline.get("backend"),
            "model": timeline.get("model"),
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required_outputs(include_manifest=False)
        },
        "notebooks": [
            "ultralytics_yolo26x_step_by_step.ipynb",
            "ultralytics_yolo26x_hands_on.ipynb",
        ],
    }
    temporary = MANIFEST.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)
    return manifest


def run_workflow(force: bool = False) -> dict[str, object]:
    env.require_files(
        env.SOURCE_VIDEO,
        env.YOLO_CHECKPOINT,
        env.YOLO_ONNX,
        env.QWEN_GGUF,
        env.QWEN_MMPROJ,
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    outputs_complete = all(path.is_file() for path in required_outputs(False))
    identity_ready, identity_detail = manifest_matches_current()
    reuse = outputs_complete and identity_ready and not force
    pipeline_seconds = None
    subtitle_seconds = None
    if not reuse:
        print(f"[workflow] Regenerating artifacts: {identity_detail}")
        pipeline_seconds = run_and_log(pipeline_command(), PIPELINE_LOG)
        subtitle_seconds = run_and_log(
            subtitle_command(force_analyze=True), SUBTITLE_LOG
        )
        write_manifest()
    result = validate_outputs()
    result.update({
        "reused": reuse,
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
    identity_ready, identity_detail = manifest_matches_current()
    if not identity_ready:
        raise RuntimeError(
            f"Saved outputs do not match the current model/runtime: {identity_detail}. "
            "Run scripts/run_pipeline.py --force."
        )
    log = PIPELINE_LOG.read_text(encoding="utf-8", errors="replace")
    timeline = json.loads(TIMELINE.read_text(encoding="utf-8"))
    source_info = video_info(env.SOURCE_VIDEO)
    yolo_info = video_info(YOLO_VIDEO)
    final_info = video_info(FINAL_VIDEO)
    if "decode: rocdecode" not in log or "encode: vaapi-direct" not in log:
        raise RuntimeError(
            "The saved vision pass did not use rocDecode + HIP/DRM PRIME VA-API"
        )
    performance = _pipeline_performance(log)
    if performance.get("fps", 0.0) < 50.0:
        raise RuntimeError(
            f"The saved vision pass is below the 50 FPS gate: {performance.get('fps')}"
        )
    if performance.get("frame_d2h_ms") != 0.0:
        raise RuntimeError(
            "The saved vision pass performed a full-frame device-to-host copy"
        )
    if "Avg CPU overlay: 0.00ms/frame (HIP overlay active)" not in log:
        raise RuntimeError("The saved vision pass did not use the HIP overlay")
    if (
        performance.get("direct_encode_submitted_frames")
        != performance.get("direct_encode_encoded_frames")
        or performance.get("direct_encode_encoded_frames") != source_info["frames"]
    ):
        raise RuntimeError("The direct encode queue did not drain every frame")
    if "MIGraphXExecutionProvider" not in log or "GPU I/O binding" not in log:
        raise RuntimeError("The saved run did not use Ultralytics MIGraphX I/O binding")
    if timeline.get("backend") != "llamacpp":
        raise RuntimeError(f"Unexpected VLM backend: {timeline.get('backend')}")
    if "Q8_0.gguf" not in timeline.get("model", ""):
        raise RuntimeError(f"Unexpected VLM model: {timeline.get('model')}")
    expected_frames = source_info["frames"]
    for label, info in (
        ("source", source_info),
        ("YOLO", yolo_info),
        ("final", final_info),
    ):
        accounting = (
            info["frames"],
            info["container_samples"],
            info["packets"],
            info["decoded_frames"],
        )
        if accounting != (expected_frames,) * 4:
            raise RuntimeError(
                f"{label} video frame accounting differs: {accounting}, "
                f"expected {(expected_frames,) * 4}"
            )
    for label, info in (("YOLO", yolo_info), ("final", final_info)):
        if (
            info["codec"] != "h264"
            or info["profile"] != "High"
            or info["pixel_format"] != "yuv420p"
        ):
            raise RuntimeError(f"Unexpected {label} encoding identity: {info}")
    return {
        "source": source_info,
        "yolo": yolo_info,
        "final": final_info,
        "segments": len(timeline.get("segments", [])),
        "backend": timeline["backend"],
        "model": timeline["model"],
        "performance": performance,
        "manifest": str(MANIFEST),
        "manifest_status": identity_detail,
        "run_dir": str(RUN_DIR),
    }
