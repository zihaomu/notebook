#!/usr/bin/env python3
"""Build deterministic Ultralytics-first workshop notebooks as nbformat JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = {
    "display_name": "Python 3 (Ultralytics YOLO26 Radeon)",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "name": "python",
    "version": "3.10",
    "mimetype": "text/x-python",
    "codemirror_mode": {"name": "ipython", "version": 3},
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "file_extension": ".py",
}


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(cell_id: str, text: str) -> dict:
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "metadata": {"id": cell_id, "language": "markdown"},
        "source": lines(text),
    }


def code(cell_id: str, text: str) -> dict:
    return {
        "id": cell_id,
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id, "language": "python"},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    ids = [cell["metadata"]["id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook cell metadata.id values must be unique")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNEL,
            "language_info": LANGUAGE_INFO,
            "package": {
                "root": "ultralytics_yolo26",
                "runtime": "Ultralytics ONNX + MIGraphX EP + OpenCV HIP + llama.cpp",
                "workshop": "YOLO26x production video on AMD Radeon",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


setup_code = r'''
import importlib
import json
import os
import sys
from pathlib import Path


NOTEBOOK_DIR = Path.cwd().resolve()
BAKED_SEED = Path(os.environ.get(
    "ULTRALYTICS_WORKSHOP_SEED_DIR", "/opt/ultralytics-yolo26/seed"
)).resolve()
BAKED_MODELS = Path(os.environ.get(
    "ULTRALYTICS_BAKED_MODEL_DIR", "/opt/ultralytics-yolo26/models"
)).resolve()


def is_package_root(path: Path) -> bool:
    return (
        (path / "scripts/notebook_env.py").is_file()
        and (path / "src").is_dir()
        and (path / "data").is_dir()
    )


def find_package_root() -> Path:
    candidates = []
    for variable in ("ULTRALYTICS_YOLO26_ROOT", "OPENCV_AMD_END2END_ROOT"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]).expanduser())
    candidates.extend([NOTEBOOK_DIR, *NOTEBOOK_DIR.parents])
    candidates.extend(Path(value) for value in sys.path if value)
    candidates.append(BAKED_SEED)

    visited = set()
    for origin in candidates:
        try:
            resolved = origin.resolve()
        except (OSError, RuntimeError):
            continue
        for candidate in (resolved, *resolved.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if is_package_root(candidate):
                return candidate
    raise FileNotFoundError(
        "Cannot find the ultralytics_yolo26 sources from the current directory "
        f"({NOTEBOOK_DIR}) or immutable seed ({BAKED_SEED})."
    )


def map_baked_models(work_dir: Path, source_dir: Path) -> tuple[Path, str]:
    required = (
        "yolo26x.pt",
        "yolo26x.onnx",
        "Qwen3-VL-8B-Instruct-Q8_0.gguf",
        "mmproj-F16.gguf",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Baked model directory is incomplete: {source_dir}; missing={missing}"
        )

    alias = work_dir / "models"
    if alias.is_symlink():
        if alias.resolve() != source_dir:
            alias.unlink()
            alias.symlink_to(source_dir, target_is_directory=True)
        return alias.resolve(), "symlink"
    if alias.exists():
        if alias.resolve() == source_dir:
            return source_dir, "existing mapping"
        local_missing = [name for name in required if not (alias / name).is_file()]
        if local_missing:
            raise RuntimeError(
                f"Cannot map baked models: {alias} is an existing real path and "
                f"does not contain the release model set (missing={local_missing}). "
                "Remove that mount/directory or set ULTRALYTICS_YOLO26_MODEL_DIR "
                "to a complete model set."
            )
        return alias.resolve(), "existing model directory"

    try:
        alias.symlink_to(source_dir, target_is_directory=True)
        return alias.resolve(), "symlink"
    except OSError as error:
        print(
            f"[bootstrap] Cannot create {alias} -> {source_dir}: {error}. "
            "Using the immutable model directory directly."
        )
        return source_dir, "direct fallback"


ROOT = find_package_root()
MODEL_DIR, MODEL_MAPPING = map_baked_models(NOTEBOOK_DIR, BAKED_MODELS)
OUTPUT_DIR = NOTEBOOK_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["ULTRALYTICS_YOLO26_ROOT"] = str(ROOT)
os.environ["ULTRALYTICS_YOLO26_MODEL_DIR"] = str(MODEL_DIR)
os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["ULTRALYTICS_MIGRAPHX_CACHE_ROOT"] = str(
    MODEL_DIR / "ort-migraphx-cache"
)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts import notebook_env as env
env = importlib.reload(env)
from scripts.model_setup import ensure_models, model_status
from scripts.notebook_helpers import frame_at, show_bgr, show_bgr_grid, video_info
from scripts.validate_runtime import validate

delivery_mode = (
    "baked/offline verification"
    if os.environ.get("ULTRALYTICS_WORKSHOP_BUNDLE") == "baked"
    else "development fallback download"
)
models = ensure_models(env.MODELS, progress=True)
runtime = validate(require_models=True, require_vlm=False)
print(json.dumps({
    "notebook_dir": str(NOTEBOOK_DIR),
    "source_root": str(ROOT),
    "model_source": str(BAKED_MODELS),
    "model_alias": str(NOTEBOOK_DIR / "models"),
    "model_mapping": MODEL_MAPPING,
    "model_delivery": delivery_mode,
    **runtime,
}, indent=2))
models
'''

step_setup_code = setup_code.replace(
    'from scripts.model_setup import ensure_models, model_status\nfrom scripts.notebook_helpers import frame_at, show_bgr, show_bgr_grid, video_info',
    'from IPython.display import Markdown, display\nfrom scripts.model_setup import ensure_models, model_status\nfrom scripts import notebook_helpers\nnotebook_helpers = importlib.reload(notebook_helpers)\nfrom scripts.notebook_helpers import (\n    capture_log,\n    frame_at,\n    show_bgr,\n    show_bgr_grid,\n    show_video,\n    video_info,\n)',
).replace(
    'models = ensure_models(env.MODELS, progress=True)\nruntime = validate(require_models=True, require_vlm=False)\nprint(json.dumps({\n    "notebook_dir": str(NOTEBOOK_DIR),\n    "source_root": str(ROOT),\n    "model_source": str(BAKED_MODELS),\n    "model_alias": str(NOTEBOOK_DIR / "models"),\n    "model_mapping": MODEL_MAPPING,\n    "model_delivery": delivery_mode,\n    **runtime,\n}, indent=2))\nmodels',
    'with capture_log(env.OUTPUT / "logs/environment.log", "Environment validation"):\n    models = ensure_models(env.MODELS, progress=True)\n    runtime = validate(require_models=True, require_vlm=False)\n\nsource_info = video_info(env.SOURCE_VIDEO)\npreview_times = [\n    0.0,\n    source_info["duration_seconds"] / 2,\n    max(0.0, source_info["duration_seconds"] - 0.2),\n]\nsource_preview_frames = [frame_at(env.SOURCE_VIDEO, seconds) for seconds in preview_times]\nfirst_frame = source_preview_frames[0]\ndisplay(Markdown(\n    f"**Runtime ready:** `{runtime[\'torch_gpu\']}` / `{runtime[\'gpu_arch\']}`; "\n    f"OpenCV HIP devices: `{runtime[\'opencv_hip_devices\']}`; model mapping: "\n    f"`{NOTEBOOK_DIR / \'models\'}` -> `{MODEL_DIR}` ({MODEL_MAPPING})."\n))\nshow_bgr_grid(\n    source_preview_frames,\n    [f"Input at {seconds:.1f} s" for seconds in preview_times],\n    columns=3,\n)',
)

_STEP_ASSET_SUMMARY = r"""
verified_model_names = [Path(item["path"]).name for item in models if item["ready"]]
display(Markdown(
    "### Verified model and backend assets\n\n"
    + "\n".join(f"- `{name}`" for name in verified_model_names)
    + "\n"
    + f"- ORT providers: `{', '.join(runtime['onnxruntime_providers'])}`\n"
    + f"- MIGraphX cache: `{runtime['migraphx_cache']}`\n"
    + f"- Writable output: `{env.OUTPUT}`"
))
"""
_STEP_NATIVE_BUILD = r"""
native_build = ROOT / "native/build"
if native_build.is_dir():
    native_build_text = str(native_build)
    if native_build_text not in sys.path:
        sys.path.insert(0, native_build_text)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = existing_pythonpath.split(":") if existing_pythonpath else []
    if native_build_text not in pythonpath_entries:
        os.environ["PYTHONPATH"] = ":".join([native_build_text, *pythonpath_entries])
"""
step_setup_code = step_setup_code.replace(
    'OUTPUT_DIR = NOTEBOOK_DIR / "output"',
    'OUTPUT_DIR = NOTEBOOK_DIR / "output" / "step_by_step"',
) + _STEP_NATIVE_BUILD + _STEP_ASSET_SUMMARY

step_cells = [
    markdown("step-title", r'''
# YOLO26x Step by Step: From PyTorch to a GPU-Resident Video Pipeline

**Speaker:** Zihao Mu, Member of Technical Staff, Product Application Engineering, AMD

This notebook keeps the released YOLO + VLM workflow unchanged. It opens the black box one layer at a time and measures where time is spent:

```text
YOLO26x .pt
  -> static ONNX
  -> Ultralytics + ONNX Runtime MIGraphX EP
  -> explicit H2D / D2H transfer lab
  -> resident GPU buffers + I/O Binding
  -> GPU preprocess + GPU NMS
  -> rocDecode + direct VA-API encode
  -> Qwen3-VL as a separately measured semantic stage
```

The goal is not to memorize one FPS number. The goal is to identify each hardware boundary, measure it with synchronization, then remove or overlap the expensive boundaries without changing model correctness.
'''),
    markdown("step-setup-md", r'''
## 1. Verify the fixed environment

The workshop image already contains the checkpoint, release ONNX, MIGraphX cache, Qwen3-VL weights, and the patched Ultralytics backend. This cell still performs the complete model/GPU/runtime validation, but keeps verbose output in `output/logs/environment.log` and presents the verified environment with real input frames.
'''),
    code("step-setup", step_setup_code),
    markdown("step-export-md", r'''
## 2. Convert the YOLO26x checkpoint from PyTorch to static ONNX

The export is deliberately static: batch 1, `640 x 640`, and an end-to-end `[1, 300, 6]` output. Static shapes let MIGraphX compile and reuse a stable program.

The checkpoint is copied into `output/export/` first, so the export cannot overwrite the immutable release model. `RUN_EXPORT=1` is the default for this teaching notebook. Set it to `0` only when reusing the verified release ONNX.
'''),
    code("step-export", r'''
import gc
import hashlib
import shutil
import time

import onnx
import torch
from ultralytics import YOLO

RUN_EXPORT = os.environ.get("RUN_EXPORT", "1") == "1"
export_dir = env.OUTPUT / "export"
export_dir.mkdir(parents=True, exist_ok=True)

with capture_log(env.OUTPUT / "logs/pt_to_onnx.log", "PyTorch inference and ONNX export"):
    pt_model = YOLO(str(env.YOLO_CHECKPOINT))
    pt_result = pt_model.predict(
        first_frame,
        device=0,
        half=True,
        imgsz=640,
        conf=0.5,
        iou=0.45,
        verbose=False,
    )[0]
    if RUN_EXPORT:
        checkpoint_copy = export_dir / env.YOLO_CHECKPOINT.name
        shutil.copy2(env.YOLO_CHECKPOINT, checkpoint_copy)
        export_started = time.perf_counter()
        export_model = YOLO(str(checkpoint_copy))
        workshop_onnx = Path(export_model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=False,
            device=0,
        )).resolve()
        export_seconds = time.perf_counter() - export_started
        del export_model
    else:
        workshop_onnx = env.YOLO_ONNX
        export_seconds = None

pt_plot = pt_result.plot()
onnx_model = onnx.load(str(workshop_onnx), load_external_data=False)
onnx.checker.check_model(onnx_model)
input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
metadata = {item.key: item.value for item in onnx_model.metadata_props}
onnx_sha256 = hashlib.sha256(workshop_onnx.read_bytes()).hexdigest()
export_record = {
    "checkpoint": str(env.YOLO_CHECKPOINT),
    "onnx": str(workshop_onnx),
    "export_seconds": None if export_seconds is None else round(export_seconds, 3),
    "bytes": workshop_onnx.stat().st_size,
    "sha256": onnx_sha256,
    "input_shape": input_shape,
    "output_shape": output_shape,
    "dynamic": False,
    "task": metadata.get("task"),
    "end2end": metadata.get("end2end"),
}
export_summary = (
    f"exported in `{export_seconds:.2f} s`"
    if export_seconds is not None
    else "reused the verified release ONNX"
)
display(Markdown(
    f"**Conversion result:** `{input_shape}` -> `{output_shape}`; "
    f"{workshop_onnx.stat().st_size / 1024**2:.1f} MiB; {export_summary}."
))
show_bgr_grid(
    [first_frame, pt_plot],
    ["Input frame", f"PyTorch checkpoint output ({len(pt_result.boxes)} boxes)"],
    columns=2,
)
assert input_shape == [1, 3, 640, 640]
assert output_shape == [1, 300, 6]
del pt_result, pt_model
gc.collect()
torch.cuda.empty_cache()
'''),
    markdown("step-predict-md", r'''
## 3. Run ONNX through the Ultralytics MIGraphX backend

Ultralytics still owns model loading and `predict()`. The workshop fork selects `MIGraphXExecutionProvider`, enables FP16, and uses GPU I/O Binding.

This cell reports three different costs:

- model object construction;
- first prediction, which includes backend initialization and cache work;
- steady high-level `predict()` latency, which includes image preprocessing, inference, postprocessing, and result construction.

By default, the ONNX produced in the previous cell runs through MIGraphX with a writable cache under `output/export/`. Set `USE_FRESH_EXPORT=0` to use the hash-verified release ONNX and its prepared cache instead.
'''),
    code("step-predict", r'''
import gc
import math
import statistics

import matplotlib.pyplot as plt
import torch

from migraphx_cache import prepare_cache

USE_FRESH_EXPORT = os.environ.get(
    "USE_FRESH_EXPORT", "1" if RUN_EXPORT else "0"
) == "1"
deployment_onnx = workshop_onnx if USE_FRESH_EXPORT else env.YOLO_ONNX
deployment_sha256 = hashlib.sha256(deployment_onnx.read_bytes()).hexdigest()
deployment_cache_root = (
    export_dir / "ort-migraphx-cache"
    if USE_FRESH_EXPORT
    else env.MIGRAPHX_CACHE
)
deployment_cache_root.mkdir(parents=True, exist_ok=True)
os.environ["ULTRALYTICS_MIGRAPHX_CACHE_ROOT"] = str(deployment_cache_root)
display(Markdown("Compiling/loading the MIGraphX program, then measuring steady-state prediction..."))

with capture_log(env.OUTPUT / "logs/onnx_migraphx.log", "ONNX/MIGraphX inference"):
    cache_dir, cache_identity = prepare_cache(
        deployment_onnx, deployment_cache_root, 0
    )
    construct_started = time.perf_counter()
    yolo = YOLO(str(deployment_onnx), task="detect")
    construct_ms = (time.perf_counter() - construct_started) * 1000

    torch.cuda.synchronize()
    first_started = time.perf_counter()
    result = yolo.predict(
        first_frame,
        device=0,
        half=True,
        imgsz=640,
        conf=0.5,
        iou=0.45,
        verbose=False,
    )[0]
    torch.cuda.synchronize()
    first_predict_ms = (time.perf_counter() - first_started) * 1000

    predict_samples_ms = []
    for _ in range(12):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = yolo.predict(
            first_frame,
            device=0,
            half=True,
            imgsz=640,
            conf=0.5,
            iou=0.45,
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        predict_samples_ms.append((time.perf_counter() - started) * 1000)

backend = yolo.predictor.model.backend
assert backend.provider == "MIGraphXExecutionProvider"
assert backend.use_io_binding and backend.migraphx_fp16
ordered_predict = sorted(predict_samples_ms)
predict_p50_ms = statistics.median(predict_samples_ms)
predict_p95_ms = ordered_predict[math.ceil(len(ordered_predict) * 0.95) - 1]
result_plot = result.plot()
backend_timing = {
    "onnx": str(deployment_onnx),
    "fresh_export_selected": USE_FRESH_EXPORT,
    "provider": backend.provider,
    "io_binding": backend.use_io_binding,
    "migraphx_fp16": backend.migraphx_fp16,
    "cache": str(cache_dir),
    "construct_ms": round(construct_ms, 3),
    "first_predict_ms": round(first_predict_ms, 3),
    "steady_predict_p50_ms": round(predict_p50_ms, 3),
    "steady_predict_p95_ms": round(predict_p95_ms, 3),
    "single_call_rate_from_p50": round(1000 / predict_p50_ms, 1),
    "boxes": len(result.boxes),
}
show_bgr_grid(
    [pt_plot, result_plot],
    ["PyTorch checkpoint", "ONNX Runtime + MIGraphX FP16"],
    columns=2,
)
figure, axis = plt.subplots(figsize=(9, 4))
labels = ["First predict\n(compile/cache)", "Steady P50", "Steady P95"]
values = [first_predict_ms, predict_p50_ms, predict_p95_ms]
bars = axis.bar(labels, values, color=["#d97706", "#007c91", "#4f6d7a"])
axis.bar_label(bars, fmt="%.1f ms", padding=3)
axis.set_ylabel("Latency (ms)")
axis.set_title("MIGraphX cold start versus steady-state prediction")
axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
plt.show()
display(Markdown(
    f"**Backend:** `{backend.provider}` / FP16 / I/O Binding. "
    f"Steady P50 `{predict_p50_ms:.2f} ms` ({1000 / predict_p50_ms:.1f} calls/s); "
    f"detected `{len(result.boxes)}` objects."
))
del result, yolo, backend
gc.collect()
torch.cuda.empty_cache()
'''),
    markdown("step-transfer-md", r'''
## 4. Measure the transfer tax: H2D and D2H

`H2D` copies host memory to GPU memory. `D2H` copies GPU memory back to the host. These synchronized measurements time one preallocated copy at a time; allocation and random-data generation are outside the timed region.

We compare:

- a full `1920 x 1080` RGB frame;
- a `1 x 3 x 640 x 640` FP32 model input;
- the compact `1 x 300 x 6` FP32 model output;
- pageable and pinned host memory.

Pinned memory can enable asynchronous DMA, but `non_blocking=True` alone does not make a dependency disappear. This cell synchronizes after every copy to measure completion rather than enqueue time. The exact numbers are machine-specific; the payload-size contrast is the transferable lesson.
'''),
    code("step-transfer", r'''
import pandas as pd


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def measure_transfer(operation, payload_bytes, repeats=50, warmup=5):
    for _ in range(warmup):
        operation()
        torch.cuda.synchronize()
    samples_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000)
    p50_ms = statistics.median(samples_ms)
    p95_ms = percentile(samples_ms, 0.95)
    return {
        "p50_ms": round(p50_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "effective_GBps": round(payload_bytes / (p50_ms * 1_000_000), 3),
        "share_of_25fps_budget_pct": round(p50_ms / 40.0 * 100, 2),
    }


transfer_cases = [
    ("1080p RGB frame", (1080, 1920, 3), torch.uint8),
    ("640x640 model input", (1, 3, 640, 640), torch.float32),
    ("300x6 compact output", (1, 300, 6), torch.float32),
]
transfer_results = []
for payload_name, shape, dtype in transfer_cases:
    pageable_source = torch.empty(shape, dtype=dtype, device="cpu")
    pinned_source = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    gpu_buffer = torch.empty(shape, dtype=dtype, device="cuda:0")
    pageable_destination = torch.empty(shape, dtype=dtype, device="cpu")
    pinned_destination = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    payload_bytes = gpu_buffer.numel() * gpu_buffer.element_size()
    operations = [
        ("H2D", "pageable", lambda: gpu_buffer.copy_(pageable_source, non_blocking=False)),
        ("H2D", "pinned", lambda: gpu_buffer.copy_(pinned_source, non_blocking=True)),
        ("D2H", "pageable", lambda: pageable_destination.copy_(gpu_buffer, non_blocking=False)),
        ("D2H", "pinned", lambda: pinned_destination.copy_(gpu_buffer, non_blocking=True)),
    ]
    for direction, host_memory, operation in operations:
        transfer_results.append({
            "payload": payload_name,
            "direction": direction,
            "host_memory": host_memory,
            "payload_MiB": round(payload_bytes / 1024**2, 3),
            **measure_transfer(operation, payload_bytes),
        })

transfer_table = pd.DataFrame(transfer_results)
figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for axis, direction in zip(axes, ("H2D", "D2H")):
    subset = transfer_table[transfer_table["direction"] == direction]
    for memory, color in (("pageable", "#d97706"), ("pinned", "#007c91")):
        values = subset[subset["host_memory"] == memory]
        positions = [index + (-0.18 if memory == "pageable" else 0.18) for index in range(len(values))]
        axis.bar(positions, values["p50_ms"], width=0.34, label=memory, color=color)
    axis.set_xticks(range(len(values)), values["payload"], rotation=18, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("P50 latency (ms, log scale)")
    axis.set_title(f"{direction}: payload size changes the cost")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
figure.tight_layout()
plt.show()
display(transfer_table[["payload", "direction", "host_memory", "payload_MiB", "p50_ms", "p95_ms"]])
'''),
    markdown("step-resident-md", r'''
## 5. Remove repeated transfers with resident GPU buffers

The production adapter initializes Ultralytics once, keeps the ORT/MIGraphX backend alive, and reuses fixed input/output allocations. rocDecode produces a DLPack GPU tensor; OpenCV HIP writes directly into reusable preprocessing buffers; I/O Binding writes the model output into a stable GPU tensor.

This cell times inference only. Compare it with the high-level `predict()` result, but do not call the difference “transfer time”: the high-level API also includes preprocessing, postprocessing, and Python result construction.
'''),
    code("step-resident", r'''
import cv2
import config as pipeline_config

pipeline_config = importlib.reload(pipeline_config)
pipeline_config.YOLO_MODEL_PATH = str(deployment_onnx)

from detector import UltralyticsYOLODetector
from postprocess import draw_detections
from preprocess import GPUPreprocessor
from video_io import RocDecodeReader

with capture_log(env.OUTPUT / "logs/gpu_resident_path.log", "GPU-resident path"):
    reader = RocDecodeReader(str(env.SOURCE_VIDEO), device_id=0)
    processor = GPUPreprocessor((640, 640), device="cuda:0")
    detector = UltralyticsYOLODetector(model_path=str(deployment_onnx), device_id=0)
    try:
        ok, rgb_gpu = reader.read_gpu()
        assert ok and rgb_gpu.is_cuda
        blob_gpu, scale, pad_w, pad_h = processor.process(rgb_gpu)
        for _ in range(5):
            raw_gpu = detector.infer_gpu(blob_gpu)
        torch.cuda.synchronize()
        resident_samples_ms = []
        input_pointer = blob_gpu.data_ptr()
        output_pointer = raw_gpu.data_ptr()
        for _ in range(30):
            torch.cuda.synchronize()
            started = time.perf_counter()
            raw_gpu = detector.infer_gpu(blob_gpu)
            torch.cuda.synchronize()
            resident_samples_ms.append((time.perf_counter() - started) * 1000)
            assert blob_gpu.data_ptr() == input_pointer
            assert raw_gpu.data_ptr() == output_pointer
        detections = detector._parse_gpu(
            raw_gpu, scale, pad_w, pad_h, tuple(rgb_gpu.shape)
        )
        decoded_rgb = rgb_gpu.cpu().numpy()
        model_input_rgb = (
            blob_gpu[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        )
    finally:
        reader.release()

resident_inference = {
    "p50_ms": round(statistics.median(resident_samples_ms), 3),
    "p95_ms": round(percentile(resident_samples_ms, 0.95), 3),
    "input_pointer": input_pointer,
    "output_pointer": output_pointer,
}
decoded_bgr = cv2.cvtColor(decoded_rgb, cv2.COLOR_RGB2BGR)
model_input_bgr = cv2.cvtColor(model_input_rgb, cv2.COLOR_RGB2BGR)
production_plot = decoded_bgr.copy()
draw_detections(production_plot, detections, names=detector.names)
show_bgr_grid(
    [decoded_bgr, model_input_bgr, production_plot],
    [
        f"rocDecode GPU frame {tuple(rgb_gpu.shape)}",
        f"OpenCV HIP letterbox {tuple(blob_gpu.shape)}",
        f"GPU NMS output ({len(detections)} boxes)",
    ],
    columns=3,
)
display(Markdown(
    f"**Resident inference:** P50 `{resident_inference['p50_ms']:.2f} ms`, "
    f"P95 `{resident_inference['p95_ms']:.2f} ms`; input/output pointers stayed stable. "
    "The two `.cpu()` copies above exist only to render this teaching figure."
))
assert blob_gpu.device.type == "cuda" and raw_gpu.device.type == "cuda"

production_detections = list(detections)
production_names = detector.names
del detector, processor, raw_gpu, blob_gpu, rgb_gpu
gc.collect()
torch.cuda.empty_cache()
'''),
    markdown("step-parity-md", r'''
## 6. Protect correctness before optimizing further

The convenience path and production path use different letterbox implementations, so byte-identical boxes are not required. They must retain the same box count and classes, with class-matched IoU above 0.90. A faster path that fails this gate is not an optimization.
'''),
    code("step-parity", r'''
from tests.test_predict_production_parity import validate as validate_parity

with capture_log(env.OUTPUT / "logs/predict_production_parity.log", "Prediction parity"):
    parity = validate_parity(deployment_onnx, env.SOURCE_VIDEO, minimum_iou=0.90)

reference_plot = result_plot.copy()
production_parity_plot = first_frame.copy()
draw_detections(production_parity_plot, production_detections, names=production_names)
show_bgr_grid(
    [reference_plot, production_parity_plot],
    [
        f"Ultralytics predict ({parity['predict_boxes']} boxes)",
        f"Production GPU path ({parity['production_boxes']} boxes)",
    ],
    columns=2,
)
figure, axis = plt.subplots(figsize=(8, 2.6))
axis.barh(
    ["Minimum class-matched IoU", "Mean class-matched IoU"],
    [parity["minimum_class_matched_iou"], parity["mean_class_matched_iou"]],
    color=["#d97706", "#007c91"],
)
axis.axvline(parity["minimum_required_iou"], color="#b91c1c", linestyle="--", label="required")
axis.set_xlim(0.8, 1.0)
axis.set_xlabel("IoU")
axis.legend()
axis.grid(axis="x", alpha=0.25)
figure.tight_layout()
plt.show()
display(Markdown(
    f"**Parity PASS:** minimum class-matched IoU "
    f"`{parity['minimum_class_matched_iou']:.4f}` >= `{parity['minimum_required_iou']:.2f}`."
))
gc.collect()
torch.cuda.empty_cache()
'''),
    markdown("step-benchmark-md", r'''
## 7. Measure every GPU-resident stage

This benchmark separates rocDecode, OpenCV HIP preprocessing, Ultralytics/MIGraphX inference, and OpenCV GPU NMS. It synchronizes every stage, reports mean/P50/P95 after warmup, and fails if any reusable pointer changes.

The benchmark excludes overlay and encoded-video transport. Its FPS is therefore a stage-boundary measurement, not the final video FPS.
'''),
    code("step-benchmark", r'''
from tests.benchmark_gpu_stages import benchmark

RUN_BENCHMARK = os.environ.get("RUN_BENCHMARK", "0") == "1"
benchmark_path = env.OUTPUT / f"benchmarks/gpu_stages_{deployment_sha256[:12]}.json"
if RUN_BENCHMARK or not benchmark_path.is_file():
    with capture_log(env.OUTPUT / "logs/gpu_stage_benchmark.log", "GPU stage benchmark"):
        gpu_benchmark = benchmark(
            env.SOURCE_VIDEO,
            frames=120,
            warmup=10,
            model_path=deployment_onnx,
        )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(gpu_benchmark, indent=2) + "\n")
else:
    gpu_benchmark = json.loads(benchmark_path.read_text())
    print(f"GPU stage benchmark: reused {benchmark_path}")

stage_names = ["decode", "preprocess", "inference", "gpu_nms"]
stage_labels = ["rocDecode", "OpenCV HIP\npreprocess", "MIGraphX\ninference", "GPU NMS"]
means = [gpu_benchmark["stages"][name]["mean_ms"] for name in stage_names]
p95s = [gpu_benchmark["stages"][name]["p95_ms"] for name in stage_names]
figure, axis = plt.subplots(figsize=(10, 4.5))
positions = range(len(stage_names))
axis.bar([value - 0.18 for value in positions], means, width=0.36, label="mean", color="#007c91")
axis.bar([value + 0.18 for value in positions], p95s, width=0.36, label="P95", color="#d97706")
axis.set_xticks(list(positions), stage_labels)
axis.set_ylabel("Latency (ms)")
axis.set_title(f"GPU-resident path: {gpu_benchmark['gpu_path_mean_ms']:.2f} ms / {gpu_benchmark['gpu_path_fps']:.1f} FPS")
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.tight_layout()
plt.show()
display(Markdown(
    f"**Stable allocations:** `{len(gpu_benchmark['stable_pointers'])}` tracked pointers; "
    f"all stayed unchanged across `{gpu_benchmark['frames']}` measured frames."
))
'''),
    markdown("step-optimization-md", r'''
## 8. Build the optimization ladder

Optimization should remove the largest repeated boundary first, then remeasure the whole relevant path:

1. **Static ONNX + MIGraphX FP16 cache** removes repeated graph interpretation and compilation.
2. **Pinned host memory** enables asynchronous DMA when a host input is unavoidable.
3. **rocDecode + DLPack** removes full-frame H2D from file decode.
4. **OpenCV HIP preprocessing** avoids a GPU-frame D2H followed by a model-input H2D.
5. **Stable buffers + ORT I/O Binding** remove model input/output round trips and allocation churn.
6. **GPU confidence filtering + NMS** reduces D2H to compact surviving detections.
7. **HIP overlay + direct VA-API** removes the full-frame D2H/raw-video pipe before encode.
8. **Sparse, separately timed VLM calls** prevent a hundreds-of-milliseconds semantic stage from defining per-frame detector throughput.

Pinned memory improves transfer mechanics; GPU residency removes the transfer. Prefer removal when the surrounding APIs allow it.
'''),
    code("step-optimization", r'''
full_frame_transfers = transfer_table[transfer_table["payload"] == "1080p RGB frame"]
pageable_h2d = full_frame_transfers[
    (full_frame_transfers["direction"] == "H2D")
    & (full_frame_transfers["host_memory"] == "pageable")
].iloc[0]
pageable_d2h = full_frame_transfers[
    (full_frame_transfers["direction"] == "D2H")
    & (full_frame_transfers["host_memory"] == "pageable")
].iloc[0]

optimization_ledger = pd.DataFrame([
    {
        "measurement": "High-level predict",
        "p50_ms": backend_timing["steady_predict_p50_ms"],
        "scope": "CPU image -> Results",
        "lesson": "Convenient baseline; includes more than inference",
    },
    {
        "measurement": "1080p H2D",
        "p50_ms": pageable_h2d["p50_ms"],
        "scope": "One full RGB frame",
        "lesson": "Remove with GPU decode",
    },
    {
        "measurement": "1080p D2H",
        "p50_ms": pageable_d2h["p50_ms"],
        "scope": "One full RGB frame",
        "lesson": "Copy compact metadata instead",
    },
    {
        "measurement": "Resident inference",
        "p50_ms": resident_inference["p50_ms"],
        "scope": "GPU input -> GPU output",
        "lesson": "Stable I/O Binding isolates execution",
    },
    {
        "measurement": "Resident stage path",
        "p50_ms": gpu_benchmark["gpu_path_mean_ms"],
        "scope": "Decode -> GPU NMS",
        "lesson": "Relevant boundary before overlay/encode",
    },
])
figure, axis = plt.subplots(figsize=(11, 4.6))
colors = ["#4f6d7a", "#d97706", "#b91c1c", "#007c91", "#2d6a4f"]
bars = axis.barh(optimization_ledger["measurement"], optimization_ledger["p50_ms"], color=colors)
axis.invert_yaxis()
axis.bar_label(bars, fmt="%.3f ms", padding=4)
axis.set_xlabel("Latency (ms)")
axis.set_title("Optimization ledger: compare boundaries before removing them")
axis.grid(axis="x", alpha=0.25)
figure.tight_layout()
plt.show()
display(optimization_ledger[["measurement", "scope", "lesson"]])
'''),
    markdown("step-vlm-md", r'''
## 9. Measure VLM input and output separately

The released workflow keeps Qwen3-VL as a separate scene-analysis pass. Every four-second segment becomes one three-frame storyboard, one HTTP request, and one caption. This is not detector-driven reasoning, and its latency must not be folded into YOLO FPS.

The cell displays exactly what the VLM received and what it returned. Set `RUN_VLM_LIVE=1` to repeat the first request against the persistent llama.cpp service; otherwise it reuses the verified timeline.
'''),
    code("step-vlm", r'''
from scripts import pipeline_workflow as workflow
workflow = importlib.reload(workflow)

RUN_VLM_LIVE = os.environ.get("RUN_VLM_LIVE", "0") == "1"
if not workflow.TIMELINE.is_file():
    with capture_log(env.OUTPUT / "logs/production_workflow.log", "Production workflow"):
        workflow_result = workflow.run_workflow(force=False)

timeline = json.loads(workflow.TIMELINE.read_text())
first_segment = timeline["segments"][0]
storyboard_path = workflow.RUN_DIR / first_segment["storyboard"]
storyboard = cv2.imread(str(storyboard_path))
assert storyboard is not None
sample_frames = [frame_at(env.SOURCE_VIDEO, seconds) for seconds in first_segment["sample_times"]]
show_bgr_grid(
    [*sample_frames, storyboard],
    [
        *[f"Sample {seconds:.1f} s" for seconds in first_segment["sample_times"]],
        "Qwen3-VL storyboard request",
    ],
    columns=2,
)

vlm_record = {
    "backend": timeline["backend"],
    "model": timeline["model"],
    "interval_seconds": timeline["interval_seconds"],
    "input": str(storyboard_path),
    "sample_times": first_segment["sample_times"],
    "output": first_segment["caption"],
    "latency_seconds": first_segment["latency_seconds"],
    "source": "verified timeline",
}
if RUN_VLM_LIVE:
    from vlm_client import LlamaCppVLMClient

    with capture_log(env.OUTPUT / "logs/vlm_live_request.log", "Live VLM request"):
        client = LlamaCppVLMClient(base_url=env.LLAMACPP_BASE_URL)
        assert client.health_check()
        started = time.perf_counter()
        live_output = client.describe_roi(storyboard, timeline["prompt"])
    vlm_record.update({
        "output": live_output,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "source": "live request",
    })
display(Markdown(
    f"**Qwen3-VL output** ({vlm_record['latency_seconds']:.3f} s, {vlm_record['source']}):  "
    f"{vlm_record['output']}"
))
'''),
    markdown("step-video-md", r'''
## 10. Connect the optimized stages to the unchanged production workflow

The production command still runs the released YOLO-only GPU path. It forces rocDecode and `vaapi-direct`: a bounded worker queue holds the resident RGB tensor, then a dedicated HIP stream performs RGB-to-NV12, box/text overlay, and submission to a DRM PRIME VA-API surface.

This final step runs or reuses the unchanged production workflow, then shows both the GPU-resident YOLO video and the rendered Qwen3-VL result. Detailed pipeline output remains in the artifact logs instead of filling the Notebook.
'''),
    code("step-video", r'''
from scripts import pipeline_workflow as workflow
workflow = importlib.reload(workflow)

RUN_PIPELINE = os.environ.get("RUN_PIPELINE", "0") == "1"
os.environ["ULTRALYTICS_MIGRAPHX_CACHE_ROOT"] = str(env.MIGRAPHX_CACHE)
outputs_ready = all(path.is_file() for path in workflow.required_outputs())
identity_ready, _ = workflow.manifest_matches_current()
if RUN_PIPELINE or not (outputs_ready and identity_ready):
    with capture_log(env.OUTPUT / "logs/production_workflow.log", "Production video workflow"):
        workflow_result = workflow.run_workflow(force=RUN_PIPELINE)
else:
    workflow_result = workflow.validate_outputs()
    print("Production video workflow: reused verified artifacts")
manifest = json.loads(workflow.MANIFEST.read_text())
performance = manifest["pipeline"]["performance"]

yolo_times = [0.0, source_info["duration_seconds"] / 2, max(0.0, source_info["duration_seconds"] - 0.2)]
yolo_frames = [frame_at(workflow.YOLO_VIDEO, seconds) for seconds in yolo_times]
final_frames = [frame_at(workflow.FINAL_VIDEO, seconds) for seconds in yolo_times]
for seconds, yolo_frame, final_frame in zip(yolo_times, yolo_frames, final_frames):
    show_bgr_grid(
        [yolo_frame, final_frame],
        [
            f"YOLO output at {seconds:.1f} s",
            f"YOLO + VLM output at {seconds:.1f} s",
        ],
        columns=1,
        size=(16, 14),
    )
show_video(workflow.YOLO_VIDEO, "YOLO26 GPU-resident detection video")
show_video(workflow.FINAL_VIDEO, "Final YOLO26 + Qwen3-VL video")
display(Markdown(
    f"**Production result:** `{performance['fps']:.1f} FPS`; full-frame D2H "
    f"`{performance['frame_d2h_ms']:.2f} ms`; submitted/encoded frames "
    f"`{performance['direct_encode_submitted_frames']}/{performance['direct_encode_encoded_frames']}`."
))
'''),
    markdown("step-audit-md", r'''
## 11. Copy audit and takeaways

### Final YOLO/no-VLM production path

```text
H.264
  -> rocDecode GPU frame
  -> OpenCV HIP preprocess
  -> Ultralytics / ORT MIGraphX FP16 with I/O Binding
  -> OpenCV GPU filtering and NMS
  -> compact detections D2H
  -> HIP overlay + RGB-to-NV12
  -> direct VA-API H.264 encode
```

There is no full-frame D2H in this path. Compact boxes/classes/scores cross to the host after GPU NMS. Notebook visualization, the JPEG/HTTP VLM input, and the subtitle renderer remain explicit host boundaries.

### Optimization method

1. Define the measurement boundary.
2. Warm up compilation and caches.
3. Synchronize asynchronous hardware before stopping a latency timer.
4. Separate model execution from preprocessing, postprocessing, and transport.
5. Remove repeated full-frame copies before micro-optimizing kernels.
6. Reuse allocations and verify pointer stability.
7. Protect every optimization with a parity or artifact-integrity gate.
8. Report P50/P95 and environment load, not only the best FPS.

**Takeaway:** moving a model to the GPU is only the first step. A fast pipeline keeps data on the right device, moves only compact results, overlaps independent work, and measures the entire boundary it claims to optimize.
'''),
]

hands_cells = [
    markdown("hands-title", r'''
# YOLO26 + VLM Hands-on: Make the Coordination Decisions

You will run the same video through three controlled experiments:

```text
YOLO-only -> VLM-only -> YOLO + asynchronous ROI VLM
```

Then you will change one supported parameter, compare evidence, and design how a future system should move from fixed polling to meaningful coordination.

**Submission:** one `decision.json`, one A/B comparison, and one six-part coordination answer.
'''),
    markdown("hands-setup-md", r'''
## 1. Map the in-image models and verify both services

The Notebook may start in any directory. The first code cell creates `./models` as a symlink to the immutable model directory inside the image, discovers the current source tree or immutable seed, and creates `./output` for this run. It never downloads models in baked mode.
'''),
    code("hands-setup", setup_code + r'''
from scripts.model_setup import wait_for_llamacpp

vlm_service = wait_for_llamacpp(timeout=60, progress=False)
print(json.dumps({
    "model_mapping": MODEL_MAPPING,
    "models": str(env.MODELS),
    "output": str(env.OUTPUT),
    "vlm_model": vlm_service["data"][0]["id"],
}, indent=2))
'''),
    markdown("hands-source-md", r'''
## 2. Inspect the video and predict the top ROIs

Before running the VLM, inspect one YOLO frame. Write down which detections you expect a confidence-ranked top-3 policy to select, and whether those are necessarily the most useful objects for scene understanding.
'''),
    code("hands-source", r'''
import gc
import time

import cv2
import pandas as pd
import torch
from IPython.display import display

from detector import UltralyticsYOLODetector
from postprocess import draw_detections
from preprocess import preprocess_frame_cpu

hands_on_dir = env.OUTPUT / "hands_on"
hands_on_dir.mkdir(parents=True, exist_ok=True)
source_info = video_info(env.SOURCE_VIDEO)
first_frame = frame_at(env.SOURCE_VIDEO, 0.0)
detector = UltralyticsYOLODetector(str(env.YOLO_ONNX), device_id=0)
blob, scale, pad_w, pad_h = preprocess_frame_cpu(first_frame)
detections = detector.detect_and_parse(blob, scale, pad_w, pad_h, first_frame.shape)
ranked_detections = sorted(detections, key=lambda item: item[4], reverse=True)
top_detections = ranked_detections[:3]

annotated = first_frame.copy()
draw_detections(annotated, detections, names=detector.names)
show_bgr(annotated, "First frame: predict which top-3 ROIs the VLM will receive")
display(pd.DataFrame(
    [
        {
            "rank": rank,
            "class": detector.names[int(det[5])],
            "confidence": round(float(det[4]), 4),
            "bbox": [round(float(value), 1) for value in det[:4]],
        }
        for rank, det in enumerate(top_detections, 1)
    ]
))

x1, y1, x2, y2 = [int(value) for value in top_detections[0][:4]]
selected_roi = first_frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)].copy()
selected_roi_path = hands_on_dir / "selected_top1_roi.jpg"
assert selected_roi.size and cv2.imwrite(str(selected_roi_path), selected_roi)
print(json.dumps({"source": source_info, "selected_roi": str(selected_roi_path)}, indent=2))

del detector
gc.collect()
torch.cuda.empty_cache()
'''),
    markdown("hands-yolo-md", r'''
## 3. Establish a fair YOLO-only baseline

This baseline deliberately uses the same CPU overlay and VA-API writer boundary as the async ROI run. It is not the faster `vaapi-direct` release benchmark. Keeping the non-VLM stages equal makes the A/B comparison meaningful.
'''),
    code("hands-yolo", r'''
from scripts.async_roi_workflow import run_experiment

RUN_HANDS_ON = os.environ.get("RUN_HANDS_ON", "0") == "1"
HANDS_ON_FRAMES = int(os.environ.get("HANDS_ON_FRAMES", "120"))
yolo_only = run_experiment(
    output_dir=hands_on_dir,
    mode="yolo-only",
    max_frames=HANDS_ON_FRAMES,
    force=RUN_HANDS_ON,
)
print(json.dumps({
    "frames": yolo_only["metrics"]["frames"],
    "fps": yolo_only["metrics"]["fps"],
    "detection_mean_ms": yolo_only["metrics"]["detection_mean_ms"],
    "frame_d2h_mean_ms": yolo_only["metrics"]["frame_d2h_mean_ms"],
    "video": yolo_only["video"],
}, indent=2))
'''),
    markdown("hands-vlm-md", r'''
## 4. Measure one VLM request before adding concurrency

The selected ROI is JPEG-encoded and sent to the persistent llama.cpp service. This client wall time includes JPEG, Base64, HTTP, vision encoding, prompt evaluation, and generation. Predict the latency before running the cell.
'''),
    code("hands-vlm", r'''
from vlm_client import LlamaCppVLMClient

vlm_client = LlamaCppVLMClient(base_url=env.LLAMACPP_BASE_URL)
assert vlm_client.health_check()
vlm_prompt = "Describe only the visible object and its immediate context in one concise sentence."
started = time.perf_counter()
vlm_only_output = vlm_client.describe_roi(selected_roi, vlm_prompt)
vlm_only_latency_ms = (time.perf_counter() - started) * 1000
vlm_only = {
    "roi": str(selected_roi_path),
    "latency_ms": round(vlm_only_latency_ms, 3),
    "output": vlm_only_output,
}
(hands_on_dir / "vlm_only.json").write_text(
    json.dumps(vlm_only, indent=2) + "\n", encoding="utf-8"
)
show_bgr(selected_roi, "VLM-only input ROI")
print(json.dumps(vlm_only, indent=2))
'''),
    markdown("hands-decision-md", r'''
## 5. Decision point: choose one executable configuration

Change only one variable from the baseline. The current implementation supports `interval` and `top_k` directly. `busy_policy=skip` and `transport=jpeg_http` are fixed implementation facts, not pretend switches.
'''),
    code("hands-decision", r'''
from scripts.async_roi_workflow import pipeline_command

# Edit one supported value, then explain your prediction before running.
DECISION = {
    "vlm_interval_frames": int(os.environ.get("HANDS_ON_INTERVAL", "30")),
    "vlm_top_k": int(os.environ.get("HANDS_ON_TOP_K", "3")),
    "busy_policy": "skip",
    "transport": "jpeg_http",
    "gpu_placement": os.environ.get("HANDS_ON_GPU_PLACEMENT", "single_gpu"),
    "max_frames": HANDS_ON_FRAMES,
}
assert DECISION["vlm_interval_frames"] in {15, 30, 60}
assert DECISION["vlm_top_k"] in {1, 3}
assert DECISION["busy_policy"] == "skip"
assert DECISION["transport"] == "jpeg_http"
assert DECISION["gpu_placement"] in {"single_gpu", "split_gpu"}
(hands_on_dir / "decision.json").write_text(
    json.dumps(DECISION, indent=2) + "\n", encoding="utf-8"
)
preview_command, _ = pipeline_command(
    output_dir=hands_on_dir,
    mode="async-roi",
    max_frames=DECISION["max_frames"],
    interval=DECISION["vlm_interval_frames"],
    top_k=DECISION["vlm_top_k"],
)
print(json.dumps(DECISION, indent=2))
print("Command:", " ".join(preview_command))
'''),
    markdown("hands-parallel-md", r'''
## 6. Run YOLO with asynchronous ROI VLM

YOLO keeps processing frames while one accepted VLM batch runs in a background thread. At each trigger opportunity the client either accepts one top-K batch or records `skipped_busy`. Async removes software waiting; it does not remove shared-GPU contention.
'''),
    code("hands-parallel", r'''
async_roi = run_experiment(
    output_dir=hands_on_dir,
    mode="async-roi",
    max_frames=DECISION["max_frames"],
    interval=DECISION["vlm_interval_frames"],
    top_k=DECISION["vlm_top_k"],
    drain_timeout=15.0,
    force=RUN_HANDS_ON,
)
print(json.dumps({
    "frames": async_roi["metrics"]["frames"],
    "fps": async_roi["metrics"]["fps"],
    "detection_mean_ms": async_roi["metrics"]["detection_mean_ms"],
    "detection_vlm_idle": async_roi["metrics"]["detection_vlm_idle"],
    "detection_vlm_active": async_roi["metrics"]["detection_vlm_active"],
    "trigger_opportunities": async_roi["metrics"]["trigger_opportunities"],
    "trigger_with_detections": async_roi["metrics"]["trigger_with_detections"],
    "vlm": async_roi["metrics"]["vlm"],
}, indent=2))
'''),
    markdown("hands-compare-md", r'''
## 7. Read the A/B speed ledger

Do not select a configuration only because it has the largest FPS. Check frame integrity, detection latency during VLM activity, skipped batches, actual completed ROIs, and semantic freshness.
'''),
    code("hands-compare", r'''
from scripts.async_roi_workflow import comparison

comparison_record = comparison(yolo_only, async_roi)
comparison_path = hands_on_dir / "comparison.json"
comparison_path.write_text(
    json.dumps(comparison_record, indent=2) + "\n", encoding="utf-8"
)
display(pd.DataFrame([
    {
        "mode": "YOLO-only",
        "frames": yolo_only["metrics"]["frames"],
        "fps": yolo_only["metrics"]["fps"],
        "detection_ms": yolo_only["metrics"]["detection_mean_ms"],
        "submitted_batches": 0,
        "skipped_busy": 0,
        "completed_rois": 0,
    },
    {
        "mode": "YOLO + async ROI VLM",
        "frames": async_roi["metrics"]["frames"],
        "fps": async_roi["metrics"]["fps"],
        "detection_ms": async_roi["metrics"]["detection_mean_ms"],
        "submitted_batches": async_roi["metrics"]["vlm"]["submitted_batches"],
        "skipped_busy": async_roi["metrics"]["vlm"]["skipped_busy"],
        "completed_rois": async_roi["metrics"]["vlm"]["completed_rois"],
    },
]))
print(json.dumps(comparison_record, indent=2))
'''),
    markdown("hands-design-md", r'''
## 8. Design challenge: turn polling into purposeful coordination

The current system polls every N frames and selects confidence-ranked ROIs. Design the next version without implementing it today. Your answer must specify Trigger, Evidence, Output, Backpressure, Failure, and Metrics.
'''),
    code("hands-design", r'''
# Edit these six answers. Keep outputs constrained and failures explicit.
DESIGN_ANSWER = {
    "trigger": os.environ.get(
        "HANDS_ON_TRIGGER",
        "Create a candidate when a stable track enters a configured interaction zone.",
    ),
    "evidence": "Send before/trigger/after full-scene frames, subject ROI, and track metadata.",
    "output": "Return one allowlisted event label, visible facts, and uncertainty as JSON.",
    "backpressure": "Keep one active request and one latest candidate; replace stale pending work.",
    "failure": "Keep deterministic YOLO/track facts and mark semantic context unavailable.",
    "metrics": "Measure event recall, duplicate suppression, queue delay, VLM latency, and YOLO active-window P95.",
}
required_answer_fields = {"trigger", "evidence", "output", "backpressure", "failure", "metrics"}
assert set(DESIGN_ANSWER) == required_answer_fields
assert all(str(value).strip() for value in DESIGN_ANSWER.values())
(hands_on_dir / "design_answer.json").write_text(
    json.dumps(DESIGN_ANSWER, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(DESIGN_ANSWER, indent=2))
'''),
    markdown("hands-takeaways-md", r'''
## 9. Takeaways

1. Async removes host-side waiting; it does not remove GPU contention.
2. `interval` controls trigger opportunities, `top_k` controls batch width, and `--parallel 3` controls server capacity.
3. Theoretical triggers are not submitted batches; submitted batches are not completed ROIs.
4. Shorter intervals improve freshness but increase busy skips and contention.
5. Fixed polling answers *how to run concurrently*; tracking and events answer *when a VLM call is worth paying for*.
'''),
]

end_cells = [
    markdown("e2e-title", r'''
# YOLO26 + Qwen3-VL End to End: Asynchronous ROI Understanding

This notebook runs the fixed workshop configuration from one video input:

```text
rocDecode -> OpenCV HIP -> Ultralytics/MIGraphX -> GPU NMS
                                      |
                         every 30 frames with detections
                                      v
                       top-3 ROI batch -> background thread
                                      v
                         llama.cpp --parallel 3
                                      |
                       latest completed descriptions
                                      v
                     CPU overlay -> VA-API encode
```

The VLM does not block the frame loop. It does share the same physical GPU by default, so the experiment measures real contention. This is fixed-interval ROI polling, not tracking or event-driven reasoning.
'''),
    markdown("e2e-setup-md", r'''
## 1. Map in-image models and verify the release pair

The Notebook can start in any writable directory. It creates `./models` as an alias to the immutable in-image model set, discovers current sources or the immutable seed, and writes all new artifacts under `./output`.
'''),
    code("e2e-setup", setup_code + r'''
from scripts.model_setup import wait_for_llamacpp

vlm_service = wait_for_llamacpp(timeout=60, progress=False)
assert "Q8_0.gguf" in vlm_service["data"][0]["id"]
print(json.dumps({
    "model_mapping": MODEL_MAPPING,
    "model_dir": str(env.MODELS),
    "output_dir": str(env.OUTPUT),
    "vlm_model": vlm_service["data"][0]["id"],
    "expected_llama_slots": 3,
}, indent=2))
'''),
    markdown("e2e-source-md", r'''
## 2. Inspect the fixed input

The reference clip contains 393 frames at 25 FPS. The async trigger has 14 theoretical opportunities: frame 1, then frames 30 through 390. Actual submitted batches will be fewer when the previous batch is still active.
'''),
    code("e2e-source", r'''
from IPython.display import Video, display

source = video_info(env.SOURCE_VIDEO)
print(json.dumps(source, indent=2))
show_bgr(frame_at(env.SOURCE_VIDEO, 0.0), "Input frame 0")
assert source["frames"] == 393 and source["fps"] == 25.0
'''),
    markdown("e2e-contract-md", r'''
## 3. Lock the asynchronous execution contract

These are implementation facts, not tuning controls in this Notebook:

- interval: 30 frames;
- batch width: at most 3 confidence-ranked ROIs;
- client: one active batch, busy triggers are skipped;
- server: 3 llama.cpp slots with prompt-cache RAM disabled;
- transport: JPEG + Base64 + HTTP;
- placement: pipeline and VLM share one physical GPU;
- output: latest completed ROI descriptions, not tracked identities.
'''),
    code("e2e-contract", r'''
from scripts.async_roi_workflow import pipeline_command

E2E_SETTINGS = {
    "max_frames": 0,
    "vlm_interval_frames": 30,
    "vlm_top_k": 3,
    "busy_policy": "skip",
    "transport": "jpeg_http",
    "llama_parallel_slots": 3,
    "gpu_placement": "single_gpu",
    "vlm_drain_timeout_seconds": 20.0,
}
e2e_dir = env.OUTPUT / "async_roi_e2e"
e2e_dir.mkdir(parents=True, exist_ok=True)
command, _ = pipeline_command(
    output_dir=e2e_dir,
    mode="async-roi",
    max_frames=E2E_SETTINGS["max_frames"],
    interval=E2E_SETTINGS["vlm_interval_frames"],
    top_k=E2E_SETTINGS["vlm_top_k"],
    drain_timeout=E2E_SETTINGS["vlm_drain_timeout_seconds"],
)
print(json.dumps(E2E_SETTINGS, indent=2))
print("Command:", " ".join(command))
'''),
    markdown("e2e-run-md", r'''
## 4. Run or reuse the complete A/B workflow

`RUN_END_TO_END=1` regenerates both videos. The YOLO-only baseline deliberately uses the same CPU overlay and VA-API writer boundary as the VLM run, so the comparison isolates VLM contention rather than mixing two different encoder paths.
'''),
    code("e2e-run", r'''
from scripts.async_roi_workflow import comparison, run_experiment, write_manifest

RUN_END_TO_END = os.environ.get("RUN_END_TO_END", "0") == "1"
yolo_only = run_experiment(
    output_dir=e2e_dir,
    mode="yolo-only",
    max_frames=E2E_SETTINGS["max_frames"],
    force=RUN_END_TO_END,
)
async_roi = run_experiment(
    output_dir=e2e_dir,
    mode="async-roi",
    max_frames=E2E_SETTINGS["max_frames"],
    interval=E2E_SETTINGS["vlm_interval_frames"],
    top_k=E2E_SETTINGS["vlm_top_k"],
    drain_timeout=E2E_SETTINGS["vlm_drain_timeout_seconds"],
    force=RUN_END_TO_END,
)
speed_ledger = comparison(yolo_only, async_roi)
(e2e_dir / "speed_ledger.json").write_text(
    json.dumps(speed_ledger, indent=2) + "\n", encoding="utf-8"
)
manifest_path = write_manifest(
    output_dir=e2e_dir,
    results={"yolo_only": yolo_only, "async_roi": async_roi},
    settings=E2E_SETTINGS,
)
print(json.dumps({
    "yolo_only_fps": speed_ledger["yolo_only_fps"],
    "parallel_fps": speed_ledger["parallel_fps"],
    "fps_retained_percent": speed_ledger["fps_retained_percent"],
    "manifest": str(manifest_path),
}, indent=2))
'''),
    markdown("e2e-video-md", r'''
## 5. Inspect the final async ROI video

The panel displays the latest completed descriptions. It does not imply identity continuity: this pipeline has no tracker, and a description originates from a prior trigger frame.
'''),
    code("e2e-video", r'''
async_video = Path(async_roi["paths"]["video"])
print(json.dumps(async_roi["video"], indent=2))
display(Video(str(async_video), embed=True, html_attributes="controls"))
'''),
    markdown("e2e-counters-md", r'''
## 6. Read actual triggers, skips, completions, and latency

Do not estimate VLM calls with `frames // interval`. A trigger also requires detections, and an active batch causes the next trigger to be skipped. The metrics below come from the actual async client.
'''),
    code("e2e-counters", r'''
import pandas as pd

parallel_metrics = async_roi["metrics"]
vlm_metrics = parallel_metrics["vlm"]
latest = pd.DataFrame(vlm_metrics["latest_descriptions"])
display(pd.DataFrame([{
    "frames": parallel_metrics["frames"],
    "trigger_opportunities": parallel_metrics["trigger_opportunities"],
    "trigger_with_detections": parallel_metrics["trigger_with_detections"],
    "submitted_batches": vlm_metrics["submitted_batches"],
    "skipped_busy": vlm_metrics["skipped_busy"],
    "submitted_rois": vlm_metrics["submitted_rois"],
    "completed_rois": vlm_metrics["completed_rois"],
    "failed_rois": vlm_metrics["failed_rois"],
    "batch_p50_ms": vlm_metrics["batch_latency_p50_ms"],
    "batch_p95_ms": vlm_metrics["batch_latency_p95_ms"],
}]))
print("Latest completed ROI descriptions:")
for item in vlm_metrics["latest_descriptions"]:
    print("-", item["description"])
assert vlm_metrics["submitted_batches"] == vlm_metrics["completed_batches"]
assert vlm_metrics["completed_rois"] > 0 and vlm_metrics["failed_rois"] == 0
'''),
    markdown("e2e-performance-md", r'''
## 7. Compare YOLO-only and shared-GPU execution

Async execution removes host-side waiting, not shared-GPU contention. Compare overall video FPS with detector latency while the VLM is idle and active. The separate drain duration is not included in video-pipeline FPS.
'''),
    code("e2e-performance", r'''
display(pd.DataFrame([
    {
        "mode": "YOLO-only",
        "frames": yolo_only["metrics"]["frames"],
        "fps": yolo_only["metrics"]["fps"],
        "detection_mean_ms": yolo_only["metrics"]["detection_mean_ms"],
        "detection_active_p95_ms": 0.0,
        "vlm_drain_seconds": 0.0,
    },
    {
        "mode": "YOLO + async ROI VLM",
        "frames": parallel_metrics["frames"],
        "fps": parallel_metrics["fps"],
        "detection_mean_ms": parallel_metrics["detection_mean_ms"],
        "detection_active_p95_ms": parallel_metrics["detection_vlm_active"]["p95_ms"],
        "vlm_drain_seconds": parallel_metrics["vlm_drain_seconds"],
    },
]))
print(json.dumps(speed_ledger, indent=2))
assert parallel_metrics["frames"] == yolo_only["metrics"]["frames"] == 393
assert parallel_metrics["fps"] >= 25.0
'''),
    markdown("e2e-audit-md", r'''
## 8. Verify frame integrity, artifacts, and the copy boundary

The current async VLM path downloads each full decoded frame for CPU overlay and JPEG ROI extraction, then the VA-API writer uploads the annotated frame. This is intentionally different from the YOLO-only `vaapi-direct` production path.
'''),
    code("e2e-audit", r'''
manifest = json.loads(manifest_path.read_text())
video = async_roi["video"]
print(json.dumps({
    "manifest_status": manifest["status"],
    "workflow": manifest["workflow"],
    "settings": manifest["settings"],
    "video": video,
    "frame_d2h_mean_ms": parallel_metrics["frame_d2h_mean_ms"],
    "artifacts": manifest["artifacts"],
}, indent=2))
assert manifest["status"] == "PASS"
assert manifest["settings"]["vlm_interval_frames"] == 30
assert manifest["settings"]["vlm_top_k"] == 3
assert manifest["settings"]["llama_parallel_slots"] == 3
assert video["container_samples"] == video["packets"] == video["decoded_frames"] == 393
assert parallel_metrics["frame_d2h_mean_ms"] > 0
'''),
    markdown("e2e-takeaways-md", r'''
## 9. Takeaways and the open coordination question

1. YOLO and VLM run concurrently in software, but compete for one GPU.
2. Thirty frames is the trigger period, top-3 is the batch width, and slot 3 is server capacity.
3. Trigger opportunities, submitted batches, and completed ROIs are different counts.
4. This path intentionally pays a full-frame host boundary for CPU overlay and JPEG transport.
5. The descriptions are latest ROI results, not tracked identities.

**Open design question:** How would tracking, event triggers, and before/after evidence replace fixed polling without blocking the frame loop?
'''),
]


def replace_cell_source(cells, cell_id, source_text):
    result = []
    for cell in cells:
        copied = dict(cell)
        copied["metadata"] = dict(cell["metadata"])
        if copied["metadata"]["id"] == cell_id:
            copied["source"] = lines(source_text)
        result.append(copied)
    return result


step_v2_cells = step_cells
step_v2_markdown = {
    'step-title': '# YOLO26x Step by Step: Follow One Frame\n\n**Question:** a warm model call is fast. Why can a video pipeline still be slow?\n\n```text\ncorrect model -> deployment engine -> frame boundaries -> resident path\n-> parity -> stage A/B -> VLM side path -> production proof\n```\n\nEach step answers one question with one artifact. Detailed logs stay in `output/step_by_step/logs/`.\n',
    'step-setup-md': '## 1. Lock the experiment\n\n**Question:** are model, GPU, provider, video, and cache fixed?\n\n**Evidence:** verified assets plus three source frames.  \n**Next:** establish a familiar PyTorch result.\n',
    'step-export-md': '## 2. Establish the model contract\n\n**Question:** what must deployment preserve?\n\n**Evidence:** PyTorch detections and static ONNX: `1x3x640x640 -> 1x300x6`.  \n**Next:** separate first-run cost from steady-state cost.\n',
    'step-predict-md': '## 3. Separate cold start from warm inference\n\n**Question:** what does MIGraphX pay once, and what repeats every frame?\n\n**Evidence:** first predict versus warm P50/P95, with PT/ONNX visual parity.  \n**Next:** model latency is not video latency; follow one frame outside the model.\n',
    'step-transfer-md': '## 4. Follow one frame across host and GPU\n\n**Question:** where does a frame cross device boundaries?\n\n```text\nCPU round-trip: GPU decode -> full-frame D2H -> CPU overlay -> upload -> GPU encode\nGPU direct:     GPU decode -> preprocess -> infer/NMS -> overlay/NV12 -> encode\n```\n\n**Evidence:** bytes, direction, synchronized P50/P95, and 25 FPS traffic.  \n**Next:** remove repeated crossings instead of only tuning DMA.\n',
    'step-resident-md': '## 5. Keep the frame resident\n\n**Question:** can decode, preprocess, inference, and NMS reuse GPU buffers?\n\n**Evidence:** GPU tensors, detections, stable pointers, inference P50/P95.  \n**Next:** prove the result did not change.\n',
    'step-parity-md': '## 6. Gate optimization with correctness\n\n**Question:** did the new data path change detections?\n\n**Pass:** same box count/classes and class-matched IoU >= 0.90.  \n**Next:** measure the resident stages.\n',
    'step-benchmark-md': '## 7. Measure the resident stages\n\n**Question:** after copies are removed, where is GPU time spent?\n\n**Evidence:** rocDecode, HIP preprocess, MIGraphX, GPU NMS; mean and P95.  \n**Next:** compare complete host round-trip and GPU-direct boundaries.\n',
    'step-optimization-md': '## 8. Prove the boundary change end to end\n\n**Question:** does deleting the host round-trip improve the real video path?\n\n**Controlled A/B:** same video, model, 120 frames, rocDecode, and VA-API.\n\n- CPU round-trip: full-frame D2H + CPU overlay + upload;\n- GPU direct: resident overlay/NV12 + direct VA-API.\n\n**Next:** keep vision stable, then add VLM as a separate low-frequency path.\n',
    'step-vlm-md': '## 9. Keep VLM off the per-frame critical path\n\n**Question:** what does the semantic stage receive and return?\n\n**Evidence:** three-frame storyboard, one caption, one separately timed request.  \n**Rule:** do not fold VLM latency into detector FPS.\n',
    'step-video-md': '## 10. Validate production artifacts\n\n**Question:** did the optimized design survive the complete workflow?\n\n**Evidence:** YOLO and YOLO+VLM videos, 393/393 frames, zero full-frame D2H on the direct path.\n',
    'step-audit-md': '## 11. Final boundary audit\n\n```text\nH.264 -> rocDecode -> HIP preprocess -> MIGraphX -> GPU NMS\n      -> compact detections D2H\n      -> HIP overlay/RGB-to-NV12 -> direct VA-API\n```\n\n- No full-frame D2H in the YOLO direct path.\n- Compact detection metadata still crosses to host.\n- Notebook images, JPEG/HTTP VLM input, and subtitle rendering are explicit host boundaries.\n\n**Method:** locate, measure, remove, verify parity, remeasure end to end.\n',
}
for cell_id, markdown_text in step_v2_markdown.items():
    step_v2_cells = replace_cell_source(step_v2_cells, cell_id, markdown_text)
step_v2_setup = ''.join(next(cell['source'] for cell in step_v2_cells if cell['metadata']['id'] == 'step-setup'))
step_v2_setup = step_v2_setup.replace(
    'os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(OUTPUT_DIR)\n',
    'os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(OUTPUT_DIR)\n'
    'os.environ["ULTRALYTICS_YOLO26_PIPELINE_DIR"] = str(\n'
    '    NOTEBOOK_DIR / "output" / "pipeline"\n'
    ')\n',
)
step_v2_cells = replace_cell_source(step_v2_cells, 'step-setup', step_v2_setup)
step_v2_cells = replace_cell_source(step_v2_cells, 'step-transfer', 'import pandas as pd\n\n\ndef percentile(values, fraction):\n    ordered = sorted(values)\n    return ordered[math.ceil(len(ordered) * fraction) - 1]\n\n\ndef measure_transfer(operation, payload_bytes, repeats=50, warmup=5):\n    for _ in range(warmup):\n        operation()\n        torch.cuda.synchronize()\n    samples_ms = []\n    for _ in range(repeats):\n        torch.cuda.synchronize()\n        started = time.perf_counter()\n        operation()\n        torch.cuda.synchronize()\n        samples_ms.append((time.perf_counter() - started) * 1000)\n    p50_ms = statistics.median(samples_ms)\n    return {\n        "p50_ms": round(p50_ms, 4),\n        "p95_ms": round(percentile(samples_ms, 0.95), 4),\n        "effective_GBps": round(payload_bytes / (p50_ms * 1_000_000), 3),\n    }\n\n\ntransfer_cases = [\n    ("1080p RGB frame", (1080, 1920, 3), torch.uint8),\n    ("640x640 model input", (1, 3, 640, 640), torch.float32),\n    ("300x6 compact output", (1, 300, 6), torch.float32),\n]\ntransfer_results = []\nfor payload_name, shape, dtype in transfer_cases:\n    pageable_source = torch.empty(shape, dtype=dtype, device="cpu")\n    pinned_source = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)\n    gpu_buffer = torch.empty(shape, dtype=dtype, device="cuda:0")\n    pageable_destination = torch.empty(shape, dtype=dtype, device="cpu")\n    pinned_destination = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)\n    payload_bytes = gpu_buffer.numel() * gpu_buffer.element_size()\n    operations = [\n        ("H2D", "pageable", lambda: gpu_buffer.copy_(pageable_source, non_blocking=False)),\n        ("H2D", "pinned", lambda: gpu_buffer.copy_(pinned_source, non_blocking=True)),\n        ("D2H", "pageable", lambda: pageable_destination.copy_(gpu_buffer, non_blocking=False)),\n        ("D2H", "pinned", lambda: pinned_destination.copy_(gpu_buffer, non_blocking=True)),\n    ]\n    for direction, host_memory, operation in operations:\n        transfer_results.append({\n            "payload": payload_name,\n            "direction": direction,\n            "host_memory": host_memory,\n            "payload_MiB": round(payload_bytes / 1024**2, 3),\n            **measure_transfer(operation, payload_bytes),\n        })\n\ntransfer_table = pd.DataFrame(transfer_results)\nfull_frame_mib = 1080 * 1920 * 3 / 1024**2\ntraffic = pd.DataFrame([\n    {"path": "one full-frame direction", "MiB/frame": full_frame_mib, "MiB/s at 25 FPS": full_frame_mib * 25},\n    {"path": "D2H + upload round-trip", "MiB/frame": full_frame_mib * 2, "MiB/s at 25 FPS": full_frame_mib * 50},\n    {"path": "300x6 compact output", "MiB/frame": 300 * 6 * 4 / 1024**2, "MiB/s at 25 FPS": 300 * 6 * 4 / 1024**2 * 25},\n])\n\nfigure, axes = plt.subplots(1, 2, figsize=(14, 4.5))\nfor axis, direction in zip(axes, ("H2D", "D2H")):\n    subset = transfer_table[transfer_table["direction"] == direction]\n    for memory, color in (("pageable", "#d97706"), ("pinned", "#007c91")):\n        values = subset[subset["host_memory"] == memory]\n        offset = -0.18 if memory == "pageable" else 0.18\n        axis.bar([index + offset for index in range(len(values))], values["p50_ms"], width=0.34, label=memory, color=color)\n    axis.set_xticks(range(len(values)), values["payload"], rotation=18, ha="right")\n    axis.set_yscale("log")\n    axis.set_ylabel("P50 copy latency (ms, log scale)")\n    axis.set_title(direction)\n    axis.grid(axis="y", alpha=0.25)\n    axis.legend()\nfigure.tight_layout()\nplt.show()\ndisplay(traffic.round(3))\ndisplay(Markdown(\n    "**Interpretation:** one DMA is small. The expensive design is a repeated "\n    "full-frame round-trip that adds synchronization, CPU work, and another upload."\n))\n')
step_v2_cells = replace_cell_source(step_v2_cells, 'step-optimization', 'import subprocess\n\nfrom scripts.async_roi_workflow import run_experiment\nfrom scripts import pipeline_workflow as workflow\nworkflow = importlib.reload(workflow)\n\nAB_FRAMES = int(os.environ.get("STEP_V2_AB_FRAMES", "120"))\nRUN_BOUNDARY_AB = os.environ.get("RUN_BOUNDARY_AB", "0") == "1"\nab_dir = env.OUTPUT / "boundary_ab"\nab_dir.mkdir(parents=True, exist_ok=True)\n\ncpu_roundtrip = run_experiment(\n    output_dir=ab_dir,\n    mode="yolo-only",\n    max_frames=AB_FRAMES,\n    force=RUN_BOUNDARY_AB,\n)\n\ndirect_video = ab_dir / "gpu_direct.mp4"\ndirect_metrics_path = ab_dir / "gpu_direct_metrics.json"\ndirect_log = ab_dir / "gpu_direct.log"\ndirect_command = workflow.pipeline_command(\n    source=env.SOURCE_VIDEO,\n    output=direct_video,\n    max_frames=AB_FRAMES,\n)\ndirect_command.extend(["--metrics-json", str(direct_metrics_path)])\nif RUN_BOUNDARY_AB or not all(path.is_file() for path in (direct_video, direct_metrics_path, direct_log)):\n    process = subprocess.run(\n        direct_command,\n        cwd=env.ROOT,\n        env=os.environ.copy(),\n        capture_output=True,\n        text=True,\n    )\n    direct_log.write_text(process.stdout + process.stderr, encoding="utf-8")\n    if process.returncode:\n        raise RuntimeError(f"GPU-direct run failed; see {direct_log}")\nelse:\n    print(f"GPU-direct boundary run: reused {direct_metrics_path}")\n\ndirect_metrics = json.loads(direct_metrics_path.read_text())\ndirect_video_info = workflow.video_info(direct_video)\nfor key in ("container_samples", "packets", "decoded_frames"):\n    assert direct_video_info[key] == AB_FRAMES, (key, direct_video_info[key])\nassert cpu_roundtrip["metrics"]["frames"] == direct_metrics["frames"] == AB_FRAMES\nassert cpu_roundtrip["metrics"]["frame_d2h_mean_ms"] > 0\nassert direct_metrics["frame_d2h_mean_ms"] == 0\n\nboundary_ab = pd.DataFrame([\n    {\n        "path": "CPU round-trip",\n        "fps": cpu_roundtrip["metrics"]["fps"],\n        "full_frame_d2h_ms": cpu_roundtrip["metrics"]["frame_d2h_mean_ms"],\n        "overlay_ms": cpu_roundtrip["metrics"]["overlay_mean_ms"],\n        "encode_feed_ms": cpu_roundtrip["metrics"]["encode_feed_mean_ms"],\n    },\n    {\n        "path": "GPU direct",\n        "fps": direct_metrics["fps"],\n        "full_frame_d2h_ms": direct_metrics["frame_d2h_mean_ms"],\n        "overlay_ms": direct_metrics["overlay_mean_ms"],\n        "encode_feed_ms": direct_metrics["encode_feed_mean_ms"],\n    },\n])\nfigure, axes = plt.subplots(1, 2, figsize=(12, 4.2))\naxes[0].bar(boundary_ab["path"], boundary_ab["fps"], color=["#d97706", "#007c91"])\naxes[0].set_ylabel("Video pipeline FPS")\naxes[0].set_title("Same 120 frames, different boundary")\naxes[0].grid(axis="y", alpha=0.25)\naxes[1].bar(boundary_ab["path"], boundary_ab["full_frame_d2h_ms"], color=["#d97706", "#007c91"])\naxes[1].set_ylabel("Full-frame D2H (ms/frame)")\naxes[1].set_title("Delete the round-trip, not just the DMA")\naxes[1].grid(axis="y", alpha=0.25)\nfigure.tight_layout()\nplt.show()\ndisplay(boundary_ab.round(3))\ndisplay(Markdown(\n    f"**Result:** GPU direct is `{direct_metrics[\'fps\'] / cpu_roundtrip[\'metrics\'][\'fps\']:.2f}x` "\n    "the CPU round-trip throughput, with zero full-frame D2H."\n))\n')
step_v2_cells = [
    {
        **cell,
        'id': cell['id'].replace('step-', 'step-v2-', 1),
        'metadata': {
            **cell['metadata'],
            'id': cell['metadata']['id'].replace('step-', 'step-v2-', 1),
        },
    }
    for cell in step_v2_cells
]

def hidden_code(cell_id, text):
    cell = code(cell_id, text)
    cell["metadata"]["jupyter"] = {"source_hidden": True}
    cell["metadata"]["tags"] = ["hide-input"]
    return cell


hands_v2_setup = setup_code.replace(
    'OUTPUT_DIR = NOTEBOOK_DIR / "output"',
    'OUTPUT_DIR = NOTEBOOK_DIR / "output" / "hands_on"',
).replace(
    'os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(OUTPUT_DIR)\n',
    'os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(OUTPUT_DIR)\n'
    'os.environ["ULTRALYTICS_YOLO26_PIPELINE_DIR"] = str(\n'
    '    NOTEBOOK_DIR / "output" / "pipeline"\n'
    ')\n',
).replace(
    'from scripts.model_setup import ensure_models, model_status\nfrom scripts.notebook_helpers import frame_at, show_bgr, show_bgr_grid, video_info',
    'from IPython.display import Markdown, display\nimport cv2\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scripts.model_setup import ensure_models, model_status, wait_for_llamacpp\nfrom scripts.notebook_helpers import capture_log, frame_at, show_bgr, show_bgr_grid, show_video, video_info',
).replace(
    'models = ensure_models(env.MODELS, progress=True)\nruntime = validate(require_models=True, require_vlm=False)\nprint(json.dumps({\n    "notebook_dir": str(NOTEBOOK_DIR),\n    "source_root": str(ROOT),\n    "model_source": str(BAKED_MODELS),\n    "model_alias": str(NOTEBOOK_DIR / "models"),\n    "model_mapping": MODEL_MAPPING,\n    "model_delivery": delivery_mode,\n    **runtime,\n}, indent=2))\nmodels',
    'from scripts.hands_on_prompt_lab import DEFAULT_QUESTION, PromptLab, compose_prompt\nwith capture_log(env.OUTPUT / "logs/environment.log", "Environment validation"):\n    models = ensure_models(env.MODELS, progress=True)\n    runtime = validate(require_models=True, require_vlm=True)\n    wait_for_llamacpp(timeout=60, progress=False)\n    lab = PromptLab.prepare(output_dir=env.OUTPUT)\ndisplay(Markdown(\n    f"**Ready:** YOLO runs every frame; VLM answers at selected evidence windows. "\n    f"GPU `{runtime[\'torch_gpu\']}`; output `{env.OUTPUT}`."\n))',
)

hands_v2_cells = [
    markdown("hands-v2-title", r'''
# YOLO + VLM Hands-on: Same Video, Different Questions

Change **one line**. Keep the video and visual evidence fixed. Observe how a new question changes the VLM answer.

```text
YOLO every frame -> Coordinator selects evidence -> VLM answers your question
```
'''),
    markdown("hands-v2-model-md", r'''
## 1. Two clocks, one coordinator

| Layer | Rhythm | Controls |
|---|---|---|
| YOLO | every frame | visible objects and positions |
| Coordinator | selected moments | trigger and evidence |
| VLM | on demand | interpretation of that evidence |

You control the **question**. The application owns model loading, HTTP, timing, and rendering.
'''),
    hidden_code("hands-v2-setup", hands_v2_setup),
    markdown("hands-v2-evidence-md", r'''
## 2. Freeze the evidence

The next comparison keeps the YOLO frame, three-frame storyboard, and evidence hash identical.

Before running: what is visible, and what remains uncertain?
'''),
    hidden_code("hands-v2-evidence", r'''
evidence = lab.evidence(0)
yolo_frame = frame_at(lab.yolo_video, evidence["sample_times"][1])
storyboard = cv2.imread(evidence["storyboard"])
assert storyboard is not None
show_bgr_grid(
    [yolo_frame, storyboard],
    ["YOLO: objects every frame", "VLM: three frames from one trigger"],
    columns=1,
    size=(15, 12),
)
figure, axis = plt.subplots(figsize=(12, 2.4))
axis.scatter(range(16), [1] * 16, color="#007c91", s=34, label="YOLO: every frame")
trigger_seconds = [0, 4, 8, 12]
axis.scatter(trigger_seconds, [0] * 4, color="#d97706", marker="^", s=100, label="VLM trigger")
axis.set_yticks([0, 1], ["VLM", "YOLO"])
axis.set_xlabel("Video time (seconds)")
axis.set_title("Two clocks: fast detection, slower semantic questions")
axis.legend(loc="upper right")
axis.grid(axis="x", alpha=0.2)
figure.tight_layout()
plt.show()
display(Markdown(
    f"**Trigger window:** `{evidence['start']:.1f}-{evidence['end']:.1f} s`  \n"
    f"**Evidence SHA:** `{evidence['sha256'][:16]}...`"
))
'''),
    markdown("hands-v2-card-md", r'''
## 3. Your Prompt Card

The system keeps role, length, and grounding rules fixed. Change only the question.

Examples:

- `What is happening in this scene?`
- `What road-safety risks are visible?`
- `Describe this scene for someone who cannot see it.`
'''),
    code("hands-v2-question", r'''
# EDIT ONLY THIS LINE
MY_QUESTION = "What road-safety risks are visible?"

display(Markdown(f"### Your question\n\n> {MY_QUESTION}"))
'''),
    markdown("hands-v2-predict-md", r'''
## 4. Predict before running

The evidence will not change. Which visible facts should your question emphasize?
'''),
    hidden_code("hands-v2-compare", r'''
with capture_log(env.OUTPUT / "logs/prompt_compare.log", "Prompt comparison"):
    comparison = lab.compare(MY_QUESTION)
assert comparison["evidence"]["sha256"] == evidence["sha256"]
display(Markdown(
    f"### Default question\n> {comparison['default']['question']}\n\n"
    f"**Answer:** {comparison['default']['answer']}\n\n"
    f"### Your question\n> {comparison['custom']['question']}\n\n"
    f"**Answer:** {comparison['custom']['answer']}\n\n"
    f"**Same evidence SHA:** `{comparison['evidence']['sha256'][:16]}...`"
))
'''),
    markdown("hands-v2-explain-md", r'''
## 5. Why did the answer change?

- Same evidence, different prompt -> different **focus**.
- Prompt controls the question and answer format.
- Prompt cannot add objects or events that are not visible.

Check relevance, visible grounding, and the one-sentence format.
'''),
    hidden_code("hands-v2-timeline", r'''
with capture_log(env.OUTPUT / "logs/custom_timeline.log", "Four-trigger timeline"):
    custom_timeline = lab.run_timeline(
        MY_QUESTION,
        comparison=comparison,
        force=os.environ.get("RUN_CUSTOM_TIMELINE", "0") == "1",
    )
timeline_rows = [
    {
        "trigger": f"{segment['start']:.1f}-{segment['end']:.1f} s",
        "evidence": segment["evidence_sha256"][:12],
        "answer": segment["caption"],
    }
    for segment in custom_timeline["segments"]
]
display(pd.DataFrame(timeline_rows))
'''),
    markdown("hands-v2-video-md", r'''
## 6. Same YOLO video, different semantic view

YOLO boxes stay fixed. Trigger selects the four evidence windows; your question changes the semantic timeline.
'''),
    hidden_code("hands-v2-videos", r'''
videos = lab.video_paths()
show_video(videos["baseline"], "Baseline: general scene summary")
show_video(videos["custom"], f"Custom: {MY_QUESTION}")
assert video_info(videos["baseline"])["frames"] == 393
assert video_info(videos["custom"])["frames"] == 393
'''),
    markdown("hands-v2-boundary-md", r'''
## 7. What you control

| Control | Meaning |
|---|---|
| Prompt | what the VLM focuses on |
| Trigger | when the coordinator asks |
| Evidence | which frames the VLM receives |
| YOLO boxes | fixed model facts in this exercise |

A prompt changes interpretation, not the underlying video.
'''),
    hidden_code("hands-v2-save", r'''
conclusion = "The question changed the focus while the visual evidence and YOLO detections stayed fixed."
submission_path = lab.save_submission(
    MY_QUESTION,
    comparison,
    custom_timeline,
    conclusion,
)
submission = json.loads(submission_path.read_text())
display(Markdown(
    f"**Saved:** `{submission_path}`  \n"
    f"**Focus changed:** `{submission['automatic_checks']['focus_changed']}`  \n"
    f"**Within 20 words:** `{submission['automatic_checks']['format_within_20_words']}`"
))
'''),
    markdown("hands-v2-takeaways-md", r'''
## 8. Takeaways

1. **YOLO sees objects every frame.**
2. **The coordinator decides when and what evidence reaches VLM.**
3. **Your prompt decides what the VLM focuses on.**

You write the question; the application owns the complex model and pipeline work.
'''),
]


NOTEBOOKS = {
    "ultralytics_yolo26x_step_by_step.ipynb": notebook(step_v2_cells),
    "ultralytics_yolo26x_hands_on.ipynb": notebook(hands_v2_cells),
    "ultralytics_yolo26x_end_to_end.ipynb": notebook(end_cells),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=tuple(NOTEBOOKS),
        help="Generate one notebook instead of rewriting all notebooks.",
    )
    args = parser.parse_args(argv)
    selected = {args.only: NOTEBOOKS[args.only]} if args.only else NOTEBOOKS

    for name, payload in selected.items():
        target = ROOT / name
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {target.relative_to(ROOT)} ({len(payload['cells'])} cells)")


if __name__ == "__main__":
    main()
