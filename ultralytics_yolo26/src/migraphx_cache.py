"""Target-specific cache identity for Ultralytics ONNX + MIGraphX inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_identity(model_path: str | Path, device_id: int = 0) -> dict[str, object]:
    import migraphx
    import onnxruntime as ort
    import torch
    import ultralytics

    properties = torch.cuda.get_device_properties(device_id)
    return {
        "schema_version": 1,
        "onnx_file": Path(model_path).name,
        "onnx_sha256": sha256(model_path),
        "gpu_arch": str(properties.gcnArchName).split(":", 1)[0],
        "gpu_name": torch.cuda.get_device_name(device_id),
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "migraphx": str(getattr(migraphx, "__version__", "unknown")),
        "onnxruntime": ort.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_commit": os.environ.get("ULTRALYTICS_COMMIT", "unknown"),
        "patch_sha256": os.environ.get(
            "ULTRALYTICS_MIGRAPHX_PATCH_SHA256", "unknown"
        ),
        "provider": "MIGraphXExecutionProvider",
        "fp16": True,
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 300, 6],
    }


def prepare_cache(
    model_path: str | Path,
    cache_root: str | Path,
    device_id: int = 0,
) -> tuple[Path, dict[str, object]]:
    identity = runtime_identity(model_path, device_id)
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:20]
    cache_dir = Path(cache_root).resolve() / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "identity.json"
    existing_identity = None
    if metadata_path.is_file():
        try:
            existing_identity = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_identity = None
    if existing_identity != identity:
        temporary = metadata_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(metadata_path)
    os.environ["ULTRALYTICS_MIGRAPHX_CACHE_DIR"] = str(cache_dir)
    return cache_dir, identity
