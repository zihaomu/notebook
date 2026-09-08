# HunyuanOCR 1.5 Notebook Runtime Bundle

This directory is the writable host-side bundle for:

```text
hunyuanocr-notebook:20260828
sha256:0d4030ee0c87ca8976f36ee8a66fa811b4e3b5bebeb2632e70c1916d3044aed0
```

Mount this directory at `/hunyuanOCR_workspace/demo`. The image already contains
ROCm, PyTorch, Transformers, vLLM, JupyterLab, OpenSSH, fonts, and the exact
HunyuanOCR-1.5 model weights. This bundle supplies the notebook, Python helpers,
service scripts, sample images, and writable output directory.

## Start

Requirements on the host:

- AMD GPU device access through `/dev/kfd` and `/dev/dri`;
- host primary `UID:GID` `1001:1001`;
- an OpenSSH `authorized_keys` file;
- free Jupyter, vLLM, and SSH ports.

```bash
cd /home/zihaomu/bigssd/notebook/hunyuanOCRv1.5_notebook
SSH_PORT=22223 GPU=1 ./scripts/run.sh
```

The decoder attention backend defaults to `ROCM_ATTN`. Select either supported
vLLM ROCm backend when creating the container:

```bash
VLLM_ATTENTION_BACKEND=ROCM_ATTN SSH_PORT=22223 GPU=1 ./scripts/run.sh
VLLM_ATTENTION_BACKEND=TRITON_ATTN SSH_PORT=22223 GPU=1 ./scripts/run.sh
```

The setting applies to decoder attention. The HunyuanOCR vision encoder keeps
using its compatible `TORCH_SDPA` backend. Stop and recreate the container when
switching backends.

Port `2222` is the default. This host currently uses `22223` because another
container occupies `2222`. The launcher mounts this directory at
`/hunyuanOCR_workspace/demo`, mounts the authorized keys read-only, and prints
the Jupyter login token and SSH command.

```bash
SSH_PORT=22223 ./scripts/status.sh
./scripts/stop.sh
```

The main notebook is `hunyuan_ocr_demo.ipynb`. It intentionally ships with the
verified 15-cell execution cache so users can inspect the expected tables,
visualizations, timings, and diagnostics before rerunning it. `make check`
requires continuous execution counts, at least one saved output per code cell,
and no saved error output. Run it from top to bottom to reproduce the results;
OCR cells start or reuse vLLM inside the container, so no separate shell command
is needed.
