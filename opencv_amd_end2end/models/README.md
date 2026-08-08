# Model files

The two YOLO26x files are intentionally kept local for this initial source
push and must be placed in this directory before running the notebooks. The two
large Qwen3-VL GGUF files are downloaded on demand by the first cells.

## Required local YOLO assets

| File | Purpose | Size | SHA-256 |
|---|---|---:|---|
| `yolo26x.onnx` | YOLO26x graph | 223,209,857 bytes | `97d5165312f2d1957a24d7ae27863a98860ac88656d40fdac8f10de77999bc35` |
| `yolo26x_compiled.mxr` | MIGraphX FP16 graph for gfx1100 | 114,408,032 bytes | `ef790cfc6b403350c4c5281449811e2dc5e7c7e0d546adfa52caf66e36772d9a` |

The compiled `.mxr` artifact targets the validated W7900D/gfx1100 environment.
Recompile it for a different GPU architecture when necessary.

## Downloaded Qwen3-VL assets

Source repository:

```text
unsloth/Qwen3-VL-8B-Instruct-GGUF
```

Default mirror:

```text
https://hf-mirror.com
```

| File | Size | SHA-256 |
|---|---:|---|
| `Qwen3-VL-8B-Instruct-Q8_0.gguf` | 8,709,520,224 bytes | `cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7` |
| `mmproj-F16.gguf` | 1,159,030,336 bytes | `d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4` |

Both notebooks call `scripts.model_setup.ensure_models()`. Missing files are
written as `.part`, resumed with HTTP Range requests, displayed with a live
fixed Jupyter HTML progress bar, verified by size/GGUF header/SHA-256, and atomically renamed.

Manual equivalent:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/model_setup.py
```

Override the model directory only when needed:

```bash
export OPENCV_AMD_END2END_MODEL_DIR=/path/to/models
```
