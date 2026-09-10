# Ultralytics YOLO26x Production Video Workshop on AMD Radeon

Take an Ultralytics YOLO26x checkpoint beyond `predict()` and into a production-style video pipeline on AMD Radeon. The workshop starts with the familiar Ultralytics Python API, exports YOLO26x to ONNX, runs it through an Ultralytics-owned ONNX Runtime MIGraphX backend, and connects it to hardware video decode, OpenCV HIP preprocessing, GPU NMS, hardware encode, and Qwen3-VL scene understanding.

**Speaker:** Zihao Mu, Member of Technical Staff, Product Application Engineering, AMD

**Audience:** Ultralytics users who train, fine-tune, export, or deploy YOLO models, and vision engineers moving from notebook inference to continuous video pipelines.

**Experience:** Intermediate. Attendees should be comfortable with Python, basic YOLO inference, and Jupyter notebooks.

**Preparation:** A laptop with a modern browser. All experiments run in a preconfigured Radeon Cloud environment.

## Notebooks

1. [`ultralytics_yolo26x_step_by_step.ipynb`](ultralytics_yolo26x_step_by_step.ipynb) **(recommended start)**
   - Follows one frame through PT/ONNX, host/GPU boundaries, residency, parity, VLM, and production proof.
   - A controlled 120-frame A/B measured **48.209 FPS / 2.774 ms full-frame D2H** for CPU round-trip versus **97.959 FPS / 0 ms D2H** for GPU direct.
2. [`ultralytics_yolo26x_hands_on.ipynb`](ultralytics_yolo26x_hands_on.ipynb) **(recommended exercise)**
   - Participants edit one natural-language question and compare VLM answers over the exact same visual evidence.
   - YOLO runs every frame, the coordinator selects four evidence windows, and VLM answers on demand.
3. [`ultralytics_yolo26x_end_to_end.ipynb`](ultralytics_yolo26x_end_to_end.ipynb)
   - Runs the fixed 393-frame asynchronous ROI workflow with interval 30, top-3 detections, one active batch, busy-skip, and three llama.cpp slots.
   - Reports submitted/skipped/completed counts, active/idle detector latency, video integrity, and a reproducible manifest.

All three notebooks are generated from [`scripts/build_notebooks.py`](scripts/build_notebooks.py), executed in the workshop image, and saved with outputs. Each notebook creates a `models` alias in its current working directory that points to the immutable model directory exposed by the image; it does not assume the notebook is mounted at `/workspace`.

## Optional SSH access

The formal SSHD-enabled derivative is published from the immutable `20260911` image:

```bash
SSHD_IMAGE=crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:ultralytics-yolo26-workshop-20260911-sshd \
  bash scripts/build_sshd_image.sh

docker run -d --name yolo26-sshd \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  -p 2222:22 -p 8895:8888 \
  -v "$HOME/.ssh/id_ed25519.pub:/run/secrets/authorized_keys:ro" \
  -e SSH_AUTHORIZED_KEYS_FILE=/run/secrets/authorized_keys \
  -e JUPYTER_TOKEN=ultralytics-yolo26 \
  crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:ultralytics-yolo26-workshop-20260911-sshd

ssh -p 2222 root@127.0.0.1
```

The derivative explicitly creates writable `/app`, keeps `/bin/bash`, sets the final image user to `root`, and exposes `jupyter`/`jupyter-lab` on the standard `PATH`. `SERVICE_MODE=all` starts SSHD and Jupyter (default); `jupyter` and `sshd` run only one service. Public-key authentication is recommended. Password login is disabled unless `ROOT_PASSWORD` is explicitly supplied at runtime; no credentials are embedded in the image.

## Architecture

```text
YOLO26x checkpoint --Ultralytics export--> static ONNX [1,3,640,640] -> [1,300,6]
                                                |
video -> rocDecode -> OpenCV HIP preprocess -> Ultralytics ONNX/MIGraphX
                                                |
                                    ORT GPU I/O Binding
                                                |
                                    OpenCV GPU NMS
                                                |
                         compact detections + resident RGB frame
                                                |
                       bounded queue + dedicated HIP encode stream
                                                |
                 RGB -> NV12 + box/text overlay in a DRM PRIME surface
                                                |
                                  h264_vaapi -> MP4
```

Ultralytics owns model loading, metadata, provider selection, FP16 compilation, cache selection, and inference. OpenCV owns preprocessing and GPU NMS. The native bridge maps separate FFmpeg-owned linear NV12 VA-API encoder surfaces through DRM PRIME into HIP; it does not pass rocDecode surfaces directly to the encoder.

The no-VLM vision pass keeps every full-frame buffer on the GPU. A bounded worker queue overlaps HIP RGB-to-NV12/overlay and VA-API submission for frame N with inference for frame N+1. Only compact surviving detections cross to the host. The Qwen3-VL analysis and subtitle-rendering pass intentionally retains JPEG/HTTP and host-frame boundaries.

## Ultralytics MIGraphX Backend

The image installs the workshop fork at commit:

```text
zihaomu/ultralytics@34e213ca3ece4c18962f5bb922ec74da0c474d24
```

It also installs `onnxruntime-migraphx==1.24.2` and applies [`docker/patches/ultralytics-migraphx-iobinding.patch`](docker/patches/ultralytics-migraphx-iobinding.patch). The patch:

- enables static-shape GPU I/O Binding for `MIGraphXExecutionProvider`;
- binds ROCm PyTorch input/output pointers directly;
- keeps output allocations stable across the video loop;
- skips redundant torchvision NMS warmup for end-to-end YOLO26 exports.

This fork is a pinned workshop backend, not a claim that the changes are already part of an upstream Ultralytics release.

## Models and Cache

In the full release, the first notebook cell verifies these baked assets without network access:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `yolo26x.pt` | 118,667,365 | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |
| `yolo26x.onnx` | 223,287,479 | `88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a` |
| `Qwen3-VL-8B-Instruct-Q8_0.gguf` | 8,709,520,224 | `cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7` |
| `mmproj-F16.gguf` | 1,159,030,336 | `d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4` |

If the package is run from a development checkout rather than the full image, the same helper can download missing assets using `.part` files, bounded retries, HTTP Range resume, fixed size/SHA validation, and atomic replacement. In a baked image, a missing or corrupt asset is a hard error and never falls back to the network. See [`models/README.md`](models/README.md).

ORT-generated `.mxr` files live under `models/ort-migraphx-cache/<identity>/`. The identity includes the ONNX SHA, GPU architecture/name, ROCm/PyTorch HIP, MIGraphX, ONNX Runtime, Ultralytics commit, and workshop patch SHA. A cache is not assumed portable across these identities.

## Measured Results

Validated on an AMD Radeon PRO W7900D (`gfx1100`), ROCm 7.2, with the official release ONNX:

| Scope | Result |
|---|---:|
| Same-input native MIGraphX inference | 7.109 ms mean |
| Same-input Ultralytics/ORT I/O Binding | 7.374-7.800 ms mean; zero steady-state memory-copy records |
| GPU-resident detection benchmark | 50.5-87.7 FPS observed; strongly host-load dependent |
| Legacy host overlay + raw BGR/VA-API workflow | 37.1 FPS at host load 181.36 |
| Current async HIP overlay + direct VA-API workflow | **74.7 FPS** at host load 218.12 |
| Independent 393-frame direct-path repeat | **71.9 FPS** at host load 209.15 |
| Direct queue / full-frame D2H | 393 submitted = 393 encoded / 0.00 ms |
| Qwen3-VL temporal analysis | 4 segments |
| Final output | 393 frames, 1920x1220, 15.72 s |

The fixed asynchronous ROI End-to-end notebook was also validated on one W7900D with the pipeline and llama.cpp sharing the same GPU. Using interval 30, top-3 ROIs, one active batch, busy-skip, and three llama.cpp slots, the fair CPU-overlay/VA-API A/B measured **50.582 FPS YOLO-only** versus **37.045 FPS with async VLM**. All 393 frames, packets, and decoded frames were preserved. Fourteen trigger opportunities produced 3 submitted batches, 11 busy skips, and 9/9 successful ROI responses. Batch latency was **4.820 s P50 / 5.040 s P95**. Detection P50 while VLM was active was 24.999 ms versus 10.813 ms while idle, demonstrating that asynchronous control flow removes waiting but not shared-GPU contention. The async path has an explicit full-frame D2H boundary (1.663 ms/frame in this run) for CPU overlay and JPEG ROI transport; it is not the no-full-frame-D2H direct path. Evidence is under [`output/async_roi_e2e/`](output/async_roi_e2e/). The prompt-focused exercise saves its custom video and review JSON under [`output/hands_on/`](output/hands_on/).

The current vision pass reports 10.44 ms/frame detection, 7.85 ms/frame in the asynchronous GPU overlay/encode worker, and only 0.15 ms/frame of main-thread queue feed. The worker overlaps with inference, so stage means are not summed to derive end-to-end FPS. Both MP4 outputs contain 393 packets, 393 container samples, and 393 decodable frames.

The historical `50.8 FPS` result is not a same-run backend comparison: it was captured on August 8 with a different ONNX (`97d516...`) and a less-loaded host. On September 2, with host load average around 188-209, a back-to-back test on the same GPU/render node measured the historical native pipeline at `27.4 FPS` and the current Ultralytics/ORT pipeline at `29.8 FPS`. rocprof recorded zero memory-copy operations inside both native and ORT steady-state inference ranges. See [`doc/performance_diagnosis_CN.md`](doc/performance_diagnosis_CN.md) for the complete A/B and copy audit.

A whole-loop `rocprofv3 --memory-copy-trace` was also attempted after the direct encoder was added. Both CSV and JSON runs completed the application frames, then ROCm 7.2.1 failed during result export with `ring_buffer mmap failed with errno 22`; the JSON was truncated. Those files are not counted as copy-trace evidence. The valid zero-copy trace claim remains scoped to the 50-call `MEASURE_YOLO` inference markers; the no-full-frame-D2H direct-path claim is additionally enforced by the selected code path, runtime metric, queue accounting, frame accounting, and pixel audit.

Raw workshop evidence is in [`output/benchmarks/gpu_stages.json`](output/benchmarks/gpu_stages.json), [`output/benchmarks/performance_diagnosis.json`](output/benchmarks/performance_diagnosis.json), [`output/benchmarks/direct_encode_repeat_393.log`](output/benchmarks/direct_encode_repeat_393.log), the two explicit [`CSV`](output/benchmarks/rocprof_direct_copy_csv_failure.log) / [`JSON`](output/benchmarks/rocprof_direct_copy_json_failure.log) profiler failure logs, and [`output/pipeline/manifest.json`](output/pipeline/manifest.json).

## Build the Workshop Image

The default base is the user-provided validated image:

```text
crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/
muzihao2/work:opencv_end2end_2026_08_12
```

Build the dedicated baked image from a clean tracked worktree and provide an explicit release ID:

```bash
RELEASE_ID=ultralytics-yolo26-YYYY-MM-DD-rN \
  bash scripts/build_notebook_image.sh
```

Default output image:

```text
zihao/ultralytics-yolo26-workshop:rocm7.2.1-full
```

The current two-image release, including its human-readable tags and immutable OCI digest references, is recorded in [`release/current.env`](release/current.env). This host-side lock is intentionally excluded from the image bundle: it is written only after the final pipeline digest exists, avoiding a self-referential image digest.

A complete deployment uses two images: the pipeline/Jupyter image contains the application code and all model files, while the digest-pinned companion image provides the `llama-server` binary. The launcher reads the host checkout's release lock, then pulls and validates both images before creating runtime resources.

The full image contains the complete workshop under `/workspace`, all four model files under `/opt/ultralytics-yolo26/models`, the validated `gfx1100` MIGraphX `.mxr`, saved notebooks/results, patched Ultralytics, ORT MIGraphX, and two build-time native artifacts: the pybind HIP/DRM PRIME/VA-API encoder and the standalone surface capability probe. OpenCV HIP and rocDecode are inherited from the pinned base image.

The pipeline reads the immutable model directory directly. Since llama.cpp runs in a separate container, the launcher checksum-copies the baked model set once into the named volume `ultralytics_yolo26_models`, then mounts that volume read-only at `/models` for llama.cpp. No network download or host model bind mount is required. The model files occur only once in the image layer; `/workspace` does not contain a second copy.

OCI labels record the source commit, base image, Ultralytics patch, bridge source, bundle, all four model SHA-256 values, and the MIGraphX cache identity. `scripts/test_notebook_image.sh` runs with no network, no source/model mount, and a read-only root filesystem, proving the compiled runtime and all models are self-contained.

## Start Locally

Log in to the registries referenced by `release/current.env`, then choose the physical GPU and matching VA-API render node. The launcher pulls both immutable image references automatically; environment overrides are treated as development mode.

```bash
PIPELINE_GPU=0 \
LLAMA_GPU=0 \
VAAPI_DEVICE=/dev/dri/renderD128 \
bash scripts/start_notebook_container.sh
```

Open:

```text
http://127.0.0.1:8892/?token=ultralytics-yolo26
```

Stop the services with:

```bash
bash scripts/stop_notebook_container.sh
```

## Validation

Validate the release lock before starting or testing containers:

```bash
python3 scripts/validate_release_lock.py --check-local-images
```

```bash
# Image, provider, rocDecode, GPU I/O Binding, and one-frame direct encode
bash scripts/test_notebook_image.sh

# Model/runtime gate
python3 scripts/validate_runtime.py --require-models

# High-level predict vs production path
python3 tests/test_predict_production_parity.py \
  --model models/yolo26x.onnx --video data/sidewalk.mp4

# GPU-resident stage benchmark
python3 tests/benchmark_gpu_stages.py \
  --video data/sidewalk.mp4 --frames 120 --warmup 10

# GPU overlay + direct VA-API frame-accounting regression
python3 tests/test_gpu_overlay_encode.py \
  --video data/sidewalk.mp4 --output output/direct-test.mp4 --frames 30

# Validate saved 393-frame workflow identity and >=50 FPS gate
python3 scripts/run_pipeline.py --validate-only
```

Run these Python commands inside the workshop image or notebook container.
