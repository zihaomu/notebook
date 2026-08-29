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

Port `2222` is the default. This host currently uses `22223` because another
container occupies `2222`. The launcher mounts this directory at
`/hunyuanOCR_workspace/demo`, mounts the authorized keys read-only, and prints
the Jupyter login token and SSH command.

```bash
SSH_PORT=22223 ./scripts/status.sh
./scripts/stop.sh
```

The main notebook is `hunyuan_ocr_demo.ipynb`. Run it from top to
bottom. OCR cells start or reuse vLLM inside the container, so no separate shell
command is needed.
