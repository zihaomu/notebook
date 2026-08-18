# OpenCV AMD End-to-End Notebooks

A standalone OpenCV 5 + AMD Radeon ROCm/HIP end-to-end notebook package. It uses
the layout of `opencv_amd_gpu` as a packaging reference, while keeping all
end-to-end pipeline content in this separate folder.

## Notebooks

Run in this order:

1. **`opencv_amd_end2end_step_by_step.ipynb`**
   - environment, GPU, disk, and relative model-directory checks
   - automatic Qwen3-VL download from `hf-mirror.com` with live progress
   - rocDecode, OpenCV HIP, MIGraphX YOLO26x, GPU NMS, and Q8_0 VLM stages
2. **`opencv_amd_end2end.ipynb`**
   - the same environment/model preparation at the top
   - complete 393-frame workflow
   - Q8_0 temporal scene analysis, timeline, SRT, and final subtitle video
   - reuses verified outputs by default; set `RUN_PIPELINE=1` to regenerate

Both notebooks include saved outputs from successful W7900D runs.

## Models

Large model binaries are intentionally excluded from this initial source push.
Before running either notebook, place these two pipeline-ready YOLO26x files in
[`models/`](models/):

- `yolo26x.onnx`
- `yolo26x_compiled.mxr` (validated gfx1100/W7900D MIGraphX artifact)

[`models/README.md`](models/README.md) records their expected sizes and SHA-256
hashes. They can be published separately through Git LFS later.

The Qwen files are intentionally not committed:

- `Qwen3-VL-8B-Instruct-Q8_0.gguf` (8.11 GiB)
- `mmproj-F16.gguf` (1.08 GiB)

The first code cell in either notebook checks `./models`. Missing Qwen files are
downloaded from `unsloth/Qwen3-VL-8B-Instruct-GGUF` through
`https://hf-mirror.com`. The cell displays a single fixed Jupyter HTML progress bar, supports
`.part` resume, validates size/GGUF headers/SHA-256, and atomically installs each
file. See [`models/README.md`](models/README.md) for exact hashes.

Manual equivalent:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/model_setup.py
```

## Layout

```text
opencv_amd_end2end/
├── opencv_amd_end2end_step_by_step.ipynb
├── opencv_amd_end2end.ipynb
├── data/sidewalk.mp4
├── models/
│   ├── yolo26x.onnx
│   ├── yolo26x_compiled.mxr
│   └── README.md                 # Qwen source, size, and hashes
├── output/pipeline/              # verified videos, logs, timeline, SRT
├── docker/
│   ├── Dockerfile                # /workspace runtime image
│   ├── start-jupyter.sh          # image default command
│   └── validate-image.py         # GPU/runtime smoke test
├── scripts/
│   ├── model_setup.py            # hf-mirror download + live progress
│   ├── notebook_env.py
│   ├── pipeline_workflow.py
│   ├── validate_runtime.py
│   ├── run_pipeline.py
│   ├── build_notebook_image.sh
│   ├── test_notebook_image.sh
│   ├── start_notebook_container.sh
│   ├── stop_notebook_container.sh
│   └── build_notebooks.py
└── src/                           # pipeline implementation
```

## Notebook runtime image

The notebook image has two intentionally separate roots:

```text
/workspace/                        # this directory, writable host bind mount
├── opencv_amd_end2end*.ipynb
├── data/  models/  output/
└── docker/  scripts/  src/

/opencv_workspace                  # immutable image tooling
├── THIRD_PARTY_VERSIONS
├── third_party/opencv             # e038708, 5.x-hip
├── third_party/opencv_contrib     # 467cbc6, 5.x-hip-zerocopy
└── opencv_end2end/third_party -> ../third_party

/opt/opencv5                       # installed OpenCV 5 HIP runtime
/opt/rocm                          # ROCm, MIGraphX, rocDecode bindings
```

Build the image from this package. The script uses the validated local runtime
image as its base and supplies the two clean source repositories from the legacy
workspace as BuildKit contexts. The source trees are copied without `.git`
history; their exact commits and the base image ID are stored in image labels and
`/opencv_workspace/THIRD_PARTY_VERSIONS`.

```bash
bash scripts/build_notebook_image.sh
bash scripts/test_notebook_image.sh
```

Defaults:

- image: `zihao/opencv-amd-end2end:rocm7.2.1`
- base image: `zihao/opencv-llamacpp-q8:rocm7.2.1`
- legacy build inputs: `../../opencv_workspace/opencv_end2end/third_party`
- runtime host mount: this package -> `/workspace`

Override `PIPELINE_IMAGE`, `BASE_IMAGE`, `LEGACY_WORKSPACE`, `OPENCV_SOURCE`, or
`OPENCV_CONTRIB_SOURCE` when publishing or building on another host. The legacy
workspace is a **build input only**; it is not mounted into the running notebook
container.

The standard Q8_0 llama.cpp server remains a separate service image
(`zihao/llamacpp-q8:b9766-rocm`) on the same Docker network. This separation lets
it map the same writable `models/` directory read-only and start loading as soon
as the notebook download completes.

## Local Docker workflow

No external `MODEL_DIR` or legacy-workspace mount is required. The start script
builds the notebook image when it is missing, mounts this package exactly once at
`/workspace`, and starts Jupyter plus the companion llama.cpp service:

```bash
bash scripts/start_notebook_container.sh
```

Open on the remote host:

```text
http://127.0.0.1:8891/?token=opencv-amd-end2end
```

When connected through VS Code Remote SSH, forward remote port `8891` from the
**Ports** view. VS Code may choose another free local port; for example, this
workspace currently maps remote `8891` to local `8892`. Open the local URL shown
by VS Code:

```text
http://127.0.0.1:8892/lab?token=opencv-amd-end2end
```

The workspace setting in `/home/zihaomu/.vscode/settings.json` labels this port
as `OpenCV AMD JupyterLab`, restores the forwarding after reconnecting, and opens
the local browser automatically. Always use the local port reported by the
VS Code **Ports** view rather than assuming it remains `8892`.

If Qwen is missing, llama.cpp waits while the first notebook cell downloads the
GGUF files into the writable relative `models/` directory. It loads the model
automatically when both files are ready.

Stop the services:

```bash
bash scripts/stop_notebook_container.sh
```

Inspect the actual mount and image after startup:

```bash
docker inspect opencv_amd_end2end_notebook --format '{{json .Mounts}}'
docker exec opencv_amd_end2end_notebook cat /opencv_workspace/THIRD_PARTY_VERSIONS
```

The mount list should contain only this package at `/workspace`; `/opencv_workspace`
comes from the image layer. After model preparation, the CLI workflow is also
available inside the container:

```bash
docker exec -it opencv_amd_end2end_notebook bash
python3 scripts/validate_runtime.py --require-models --require-vlm
python3 scripts/run_pipeline.py --force
```

## Regenerate notebook JSON

```bash
python3 scripts/build_notebooks.py
```

Regeneration clears notebook outputs. Execute both notebooks again before
publishing if the Preview page should contain current results.
