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
├── scripts/
│   ├── model_setup.py            # hf-mirror download + live progress
│   ├── notebook_env.py
│   ├── pipeline_workflow.py
│   ├── validate_runtime.py
│   ├── run_pipeline.py
│   ├── start_notebook_container.sh
│   ├── stop_notebook_container.sh
│   └── build_notebooks.py
└── src/                           # pipeline implementation
```

## Local Docker workflow

No external `MODEL_DIR` is required. Start Jupyter first:

```bash
bash scripts/start_notebook_container.sh
```

Open:

```text
http://127.0.0.1:8891/?token=opencv-amd-end2end
```

If Qwen is missing, llama.cpp waits while the first notebook cell downloads the
GGUF files into the writable relative `models/` directory. It loads the model
automatically when both files are ready.

Stop the services:

```bash
bash scripts/stop_notebook_container.sh
```

After model preparation, the CLI workflow is also available:

```bash
python3 scripts/validate_runtime.py --require-models --require-vlm
python3 scripts/run_pipeline.py --force
```

## Regenerate notebook JSON

```bash
python3 scripts/build_notebooks.py
```

Regeneration clears notebook outputs. Execute both notebooks again before
publishing if the Preview page should contain current results.
