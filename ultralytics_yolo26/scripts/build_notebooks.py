#!/usr/bin/env python3
"""Build deterministic Ultralytics-first workshop notebooks as nbformat JSON."""

from __future__ import annotations

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


def find_package_root() -> Path:
    origins = [Path.cwd(), *(Path(value) for value in sys.path if value)]
    for origin in origins:
        for candidate in (origin, *origin.parents):
            if (
                (candidate / "scripts/notebook_env.py").is_file()
                and (candidate / "src").is_dir()
                and (candidate / "data").is_dir()
            ):
                return candidate.resolve()
    raise FileNotFoundError("Run this notebook from the ultralytics_yolo26 folder")


ROOT = find_package_root()
os.environ["ULTRALYTICS_YOLO26_ROOT"] = str(ROOT)
os.environ["ULTRALYTICS_YOLO26_MODEL_DIR"] = str(ROOT / "models")
os.environ["ULTRALYTICS_YOLO26_OUTPUT_DIR"] = str(ROOT / "output")
os.environ.setdefault(
    "ULTRALYTICS_MIGRAPHX_CACHE_ROOT", str(ROOT / "models/ort-migraphx-cache")
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
print(json.dumps({"model_delivery": delivery_mode, **runtime}, indent=2))
models
'''

step_cells = [
    markdown("step-title", r'''
# YOLO26x Beyond `predict()`: A Production Video Pipeline on AMD Radeon

**Speaker:** Zihao Mu, Member of Technical Staff, Product Application Engineering, AMD

Start with the Ultralytics API you already know, export YOLO26x to ONNX, then keep the complete no-VLM vision pass on the Radeon GPU:

```text
YOLO26x checkpoint -> ONNX -> Ultralytics/ORT MIGraphX -> OpenCV GPU NMS
                                                               |
rocDecode -> OpenCV HIP preprocess -> GPU I/O Binding -> compact detections
                                                               |
                                     bounded queue + dedicated HIP stream
                                                               |
                    DRM PRIME VAAPI surface: RGB -> NV12 + GPU overlay -> H.264
```

The workshop uses `YOLO.predict()` as the correctness baseline. In the continuous loop, Ultralytics still owns ONNX inference, OpenCV owns preprocessing and GPU NMS, and a small native bridge owns GPU overlay plus direct VA-API surface submission.
'''),
    markdown("step-setup-md", "## 1. Verify the baked model set and validate Radeon"),
    code("step-setup", setup_code),
    markdown("step-predict-md", r'''
## 2. Begin with the familiar `YOLO.predict()` API

The official release ONNX is end-to-end: its output is already `[1, 300, 6]`. The workshop fork selects ONNX Runtime's `MIGraphXExecutionProvider`, enables FP16 compilation, and binds GPU buffers without a NumPy round trip.
'''),
    code("step-predict", r'''
import cv2
from ultralytics import YOLO
from migraphx_cache import prepare_cache

capture = cv2.VideoCapture(str(env.SOURCE_VIDEO))
ok, first_frame = capture.read()
capture.release()
assert ok

cache_dir, cache_identity = prepare_cache(env.YOLO_ONNX, env.MIGRAPHX_CACHE, 0)
yolo = YOLO(str(env.YOLO_ONNX), task="detect")
result = yolo.predict(
    first_frame,
    device=0,
    half=True,
    imgsz=640,
    conf=0.5,
    iou=0.45,
    verbose=False,
)[0]
backend = yolo.predictor.model.backend
assert backend.provider == "MIGraphXExecutionProvider"
assert backend.use_io_binding and backend.migraphx_fp16
print({
    "boxes": len(result.boxes),
    "provider": backend.provider,
    "io_binding": backend.use_io_binding,
    "migraphx_fp16": backend.migraphx_fp16,
    "cache": str(cache_dir),
})
show_bgr(result.plot(), "Ultralytics YOLO26x predict() baseline")
'''),
    markdown("step-export-md", r'''
## 3. Export the checkpoint to static ONNX

Set `RUN_EXPORT=1` to execute the real export. The default classroom path inspects the pre-exported official release asset so every attendee can move directly to deployment. Export uses a copy of the checkpoint and never overwrites the verified release ONNX.
'''),
    code("step-export", r'''
import ast
import shutil
import onnx

RUN_EXPORT = os.environ.get("RUN_EXPORT", "0") == "1"
export_dir = env.OUTPUT / "export"
export_dir.mkdir(parents=True, exist_ok=True)
if RUN_EXPORT:
    checkpoint_copy = export_dir / env.YOLO_CHECKPOINT.name
    shutil.copy2(env.YOLO_CHECKPOINT, checkpoint_copy)
    exported_path = Path(
        YOLO(str(checkpoint_copy)).export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=False,
            device=0,
        )
    )
else:
    exported_path = env.YOLO_ONNX

onnx_model = onnx.load(str(exported_path), load_external_data=False)
input_shape = [item.dim_value for item in onnx_model.graph.input[0].type.tensor_type.shape.dim]
output_shape = [item.dim_value for item in onnx_model.graph.output[0].type.tensor_type.shape.dim]
metadata = {item.key: item.value for item in onnx_model.metadata_props}
print({
    "path": str(exported_path),
    "bytes": exported_path.stat().st_size,
    "input": input_shape,
    "output": output_shape,
    "end2end": metadata.get("end2end"),
    "task": metadata.get("task"),
})
assert input_shape == [1, 3, 640, 640]
assert output_shape == [1, 300, 6]
'''),
    markdown("step-resident-md", r'''
## 4. Move from an image call to resident video buffers

The production adapter is initialized by `YOLO(...)` and keeps the Ultralytics predictor/backend alive. rocDecode returns a DLPack GPU tensor. OpenCV HIP reuses fixed padded, HWC, and BCHW buffers before the Ultralytics backend writes into a pre-bound GPU output.
'''),
    code("step-resident", r'''
from detector import UltralyticsYOLODetector
from preprocess import GPUPreprocessor
from video_io import RocDecodeReader

reader = RocDecodeReader(str(env.SOURCE_VIDEO), device_id=0)
processor = GPUPreprocessor((640, 640), device="cuda:0")
detector = UltralyticsYOLODetector(model_path=str(env.YOLO_ONNX), device_id=0)
try:
    ok, rgb_gpu = reader.read_gpu()
    assert ok and rgb_gpu.is_cuda
    blob_gpu, scale, pad_w, pad_h = processor.process(rgb_gpu)
    raw_gpu = detector.infer_gpu(blob_gpu)
    detections = detector._parse_gpu(
        raw_gpu, scale, pad_w, pad_h, tuple(rgb_gpu.shape)
    )
finally:
    reader.release()

print(json.dumps({
    "decoded": {"shape": list(rgb_gpu.shape), "device": str(rgb_gpu.device), "pointer": rgb_gpu.data_ptr()},
    "preprocess": {**processor.pointer_info(), "shape": list(blob_gpu.shape)},
    "inference": detector.provider_info(),
    "detections": detections,
}, indent=2))
assert blob_gpu.device.type == "cuda" and raw_gpu.device.type == "cuda"
'''),
    markdown("step-parity-md", r'''
## 5. Check `predict()` to production parity

The two paths use different letterbox implementations, so byte-identical boxes are not expected. They must produce the same box count and classes, with class-matched IoU above 0.90.
'''),
    code("step-parity", r'''
from tests.test_predict_production_parity import validate as validate_parity

parity = validate_parity(env.YOLO_ONNX, env.SOURCE_VIDEO, minimum_iou=0.90)
print(json.dumps(parity, indent=2))
'''),
    markdown("step-benchmark-md", r'''
## 6. Benchmark the GPU-resident detection path

This benchmark excludes CPU overlay and encoded-video transport. It synchronizes each GPU stage and reports mean, P50, and P95 latency after warmup. It also asserts that all reusable buffer pointers stay unchanged.

A rocprof trace over 50 steady-state calls recorded zero memory-copy operations for both native MIGraphX and Ultralytics/ORT I/O Binding. The historical 50.8 FPS result is from a different date and ONNX, not a same-run backend comparison.
'''),
    code("step-benchmark", r'''
from tests.benchmark_gpu_stages import benchmark

RUN_BENCHMARK = os.environ.get("RUN_BENCHMARK", "0") == "1"
benchmark_path = env.OUTPUT / "benchmarks/gpu_stages.json"
if RUN_BENCHMARK or not benchmark_path.is_file():
    gpu_benchmark = benchmark(env.SOURCE_VIDEO, frames=120, warmup=10)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(gpu_benchmark, indent=2) + "\n")
else:
    gpu_benchmark = json.loads(benchmark_path.read_text())
print(json.dumps({
    "stages": gpu_benchmark["stages"],
    "gpu_path_mean_ms": gpu_benchmark["gpu_path_mean_ms"],
    "gpu_path_fps": gpu_benchmark["gpu_path_fps"],
    "stable_pointers": gpu_benchmark["stable_pointers"],
}, indent=2))
'''),
    markdown("step-video-md", r'''
## 7. Connect inference to the production video loop

The command below forces hardware decode and `vaapi-direct`. A bounded worker queue holds each resident RGB tensor, waits on a producer event from the inference stream, then performs RGB-to-NV12 and box/text/status overlay on a dedicated HIP stream inside a separate FFmpeg-owned DRM PRIME VA-API encoder surface. `VAAPI_DEVICE` selects the render node assigned by Radeon Cloud.
'''),
    code("step-video", r'''
from scripts import pipeline_workflow as workflow

print(" ".join(workflow.pipeline_command(max_frames=120)))
print("VA-API device:", os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128"))
'''),
    markdown("step-vlm-md", r'''
## 8. Extend detections with Qwen3-VL scene understanding

YOLO answers *what and where* for every frame. Qwen3-VL samples short temporal segments and adds *what is happening* as a timeline and subtitle track. The VLM runs asynchronously from the detector and its latency is reported separately.
'''),
    code("step-vlm", r'''
from scripts import pipeline_workflow as workflow

if workflow.TIMELINE.is_file():
    timeline = json.loads(workflow.TIMELINE.read_text())
    for segment in timeline["segments"]:
        print(f'{segment["start"]:4.1f}-{segment["end"]:4.1f}s  {segment["caption"]}')
else:
    print("Run the end-to-end notebook to generate the Qwen3-VL timeline.")
'''),
    markdown("step-audit-md", r'''
## 9. Copy audit

**Main YOLO/no-VLM pass: no full-frame D2H**

- rocDecode surface -> DLPack tensor
- fixed OpenCV HIP preprocess buffers
- fixed Ultralytics/ORT GPU input and output bindings
- OpenCV GPU confidence filtering and NMS
- resident RGB tensor held by a bounded queue
- HIP RGB-to-NV12 plus box/text/status overlay
- DRM PRIME VAAPI encoder surface -> `h264_vaapi`

**Explicit host boundaries**

- compact surviving boxes/classes/scores copied after GPU NMS
- notebook visualization downloads only the selected display frame
- Qwen3-VL baseline uses JPEG/HTTP transport
- subtitle rendering retains CPU text layout and the host-frame VA-API writer
- `--gpu-direct-encode off` retains the full-frame D2H/raw-pipe fallback

The direct bridge writes into a separate FFmpeg-owned encoder surface; it is not rocDecode-surface passthrough. The main vision pass has zero full-frame D2H, while the Qwen3-VL and subtitle pass still has documented host boundaries.


A whole-loop rocprof CSV/JSON export was attempted, but ROCm 7.2.1 failed in its result writer after the application completed (`ring_buffer mmap errno 22`), leaving unusable output. It is not counted as trace evidence. The successful 50-call `MEASURE_YOLO` inference trace remains a separate, narrower proof.
'''),
]

end_cells = [
    markdown("e2e-title", r'''
# Ultralytics YOLO26x Production Video Pipeline on AMD Radeon

This notebook runs the complete workshop workflow:

```text
video -> rocDecode -> OpenCV HIP -> Ultralytics ONNX/MIGraphX -> GPU NMS
      -> async HIP overlay -> DRM PRIME VAAPI surface -> annotated base video
      -> Qwen3-VL temporal scenes -> CPU subtitle render -> final MP4
```

The schema 3 manifest prevents outputs from another ONNX, GPU, ROCm/MIGraphX, ORT, Ultralytics commit, workshop patch, production vision source, or native bridge source from being silently reused.
'''),
    markdown("e2e-setup-md", "## 1. Verify the baked model set and runtime"),
    code("e2e-setup", setup_code),
    markdown("e2e-provider-md", "## 2. Initialize the Ultralytics-owned MIGraphX backend"),
    code("e2e-provider", r'''
from detector import UltralyticsYOLODetector

production_detector = UltralyticsYOLODetector(str(env.YOLO_ONNX), device_id=0)
provider = production_detector.provider_info()
print(json.dumps(provider, indent=2))
assert provider["owner"] == "ultralytics.YOLO"
assert provider["provider"] == "MIGraphXExecutionProvider"
assert provider["io_binding"] and provider["migraphx_fp16"]
'''),
    markdown("e2e-source-md", "## 3. Inspect the source video"),
    code("e2e-source", r'''
from IPython.display import HTML, display

source = video_info(env.SOURCE_VIDEO)
print(source)
source_url = "/files/" + env.SOURCE_VIDEO.relative_to(env.ROOT).as_posix()
display(HTML(
    f'<video controls preload="metadata" style="width:100%;max-width:960px" '
    f'src="{source_url}"></video>'
))
'''),
    markdown("e2e-run-md", r'''
## 4. Run or reuse the complete workflow

Set `RUN_PIPELINE=1` before starting the kernel to regenerate all 393 frames and force a fresh Qwen3-VL scene analysis. Otherwise the workflow validates the model/runtime identity in `manifest.json` before reusing saved artifacts.
'''),
    code("e2e-run", r'''
from scripts import pipeline_workflow as workflow

RUN_PIPELINE = os.environ.get("RUN_PIPELINE", "0") == "1"
workflow_result = workflow.run_workflow(force=RUN_PIPELINE)
print(json.dumps(workflow_result, indent=2))
'''),
    markdown("e2e-performance-md", r'''
## 5. Measure asynchronous GPU overlay and direct encode

The historical host-overlay/raw-BGR path reached 37.1 FPS and exposed the post-detection bottleneck. The current default removes full-frame D2H and overlaps the GPU overlay/VA-API worker with the next inference. The formal 393-frame workflow reached 74.7 FPS at host load 218.12; an independent repeat reached 71.9 FPS at load 209.15.

Worker latency and main-thread queue-feed latency are reported separately because the stages overlap. The saved workflow is accepted only when FPS is at least 50, full-frame D2H is zero, and all direct-encode submissions drain.
'''),
    code("e2e-performance", r'''
benchmark_path = env.OUTPUT / "benchmarks/gpu_stages.json"
gpu_benchmark = json.loads(benchmark_path.read_text())
manifest = json.loads(workflow.MANIFEST.read_text())
diagnosis = json.loads((env.OUTPUT / "benchmarks/performance_diagnosis.json").read_text())
print("GPU-resident detection stages:")
print(json.dumps(gpu_benchmark["stages"], indent=2))
print("GPU path:", gpu_benchmark["gpu_path_mean_ms"], "ms /", gpu_benchmark["gpu_path_fps"], "FPS")
performance = manifest["pipeline"]["performance"]
print("\nAsync direct annotated-video workflow:")
print(json.dumps(performance, indent=2))
print("\nHistorical residency and raw-pipe A/B evidence:")
print(json.dumps(diagnosis, indent=2))
assert performance["fps"] >= 50.0
assert performance["frame_d2h_ms"] == 0.0
assert performance["direct_encode_feed_ms"] < performance["gpu_overlay_direct_encode_worker_ms"]
'''),
    markdown("e2e-timeline-md", "## 6. Qwen3-VL temporal scene timeline"),
    code("e2e-timeline", r'''
import pandas as pd

timeline = json.loads(workflow.TIMELINE.read_text())
segments = pd.DataFrame(timeline["segments"])[
    ["index", "start", "end", "caption", "latency_seconds"]
]
display(segments)
assert timeline["backend"] == "llamacpp"
assert "Q8_0.gguf" in timeline["model"]
'''),
    markdown("e2e-final-md", "## 7. Play the final scene-aware video"),
    code("e2e-final", r'''
final_info = video_info(workflow.FINAL_VIDEO)
print(final_info)
final_url = "/files/" + workflow.FINAL_VIDEO.relative_to(env.ROOT).as_posix()
display(HTML(
    f'<video controls preload="metadata" style="width:100%;max-width:960px" '
    f'src="{final_url}"></video>'
))
assert final_info["frames"] == source["frames"]
'''),
    markdown("e2e-compare-md", "## 8. Compare source, YOLO detection, and scene understanding"),
    code("e2e-compare", r'''
images = [
    frame_at(env.SOURCE_VIDEO, 10),
    frame_at(workflow.YOLO_VIDEO, 10),
    frame_at(workflow.FINAL_VIDEO, 10),
]
show_bgr_grid(
    images,
    ["Source", "Ultralytics YOLO26x", "YOLO26x + Qwen3-VL"],
    size=(21, 6),
)
'''),
    markdown("e2e-manifest-md", "## 9. Verify the reproducible artifact manifest"),
    code("e2e-manifest", r'''
manifest_ready, manifest_detail = workflow.manifest_matches_current()
manifest = json.loads(workflow.MANIFEST.read_text())
print("manifest:", manifest_detail)
print(json.dumps({
    "schema_version": manifest["schema_version"],
    "runtime_identity": manifest["runtime_identity"],
    "models": manifest["models"],
    "artifacts": manifest["artifacts"],
}, indent=2))
assert manifest_ready and manifest["status"] == "PASS"
assert manifest["schema_version"] == 3
assert manifest["runtime_identity"]["hip_vaapi_bridge_sha256"]
'''),
]

NOTEBOOKS = {
    "ultralytics_yolo26x_step_by_step.ipynb": notebook(step_cells),
    "ultralytics_yolo26x_end_to_end.ipynb": notebook(end_cells),
}


def main() -> None:
    for name, payload in NOTEBOOKS.items():
        target = ROOT / name
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {target.relative_to(ROOT)} ({len(payload['cells'])} cells)")


if __name__ == "__main__":
    main()
