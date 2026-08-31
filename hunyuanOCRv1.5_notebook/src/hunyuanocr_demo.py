import base64
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from PIL import Image, ImageOps

ROOT = Path(os.environ.get("HUNYUANOCR_DEMO_ROOT", "/hunyuanOCR_workspace/demo"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/hunyuanOCR_workspace/models/HunyuanOCR"))
ASSETS_DIR = ROOT / "assets" / "images"
OUTPUT_DIR = ROOT / "outputs"
JSON_DIR = OUTPUT_DIR / "json"
VIS_DIR = OUTPUT_DIR / "visualizations"
LOG_DIR = OUTPUT_DIR / "logs"
RUNTIME_DIR = OUTPUT_DIR / "runtime"
MODEL_NAME = "tencent/HunyuanOCR"
BASE_IMAGE_DIGEST = "sha256:dc970503ebf22a87aab42a5c6cdb8b8cce5a42685c997b202e6ece6bdf3b1d86"
MODEL_WEIGHT_SHA256 = "632a1e082c4dd5a3284cf1ffcdba2fdaa06f435762c58c2f34aff0f3bd6c0249"
DOC_PARSE_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
NUMBER = r"[-+]?\d+(?:\.\d+)?"
COORDINATE_ONLY = re.compile(
    rf"^\s*\({NUMBER}\s*,\s*{NUMBER}\)"
    rf"(?:\s*,?\s*\({NUMBER}\s*,\s*{NUMBER}\))*\s*$"
)


def ensure_layout() -> dict[str, Path]:
    for path in (ASSETS_DIR, JSON_DIR, VIS_DIR, LOG_DIR, RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": ROOT,
        "model": MODEL_DIR,
        "assets": ASSETS_DIR,
        "outputs": OUTPUT_DIR,
        "json": JSON_DIR,
        "visualizations": VIS_DIR,
    }


def _no_proxy_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _no_proxy_opener().open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def base_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '18016')}/v1"


def _command(args: list[str], timeout: float = 30.0) -> dict[str, Any]:
    completed = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def runtime_report(include_torch_probe: bool = True) -> dict[str, Any]:
    ensure_layout()
    packages = {}
    for name in (
        "torch", "transformers", "vllm", "openai", "pillow", "requests",
        "matplotlib", "pandas", "jupyterlab", "ipykernel", "ipywidgets",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    os_release = {}
    release_path = Path("/etc/os-release")
    if release_path.is_file():
        for line in release_path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip('"')
    paths = {
        "demo_root": str(ROOT),
        "bundle_root": str(ROOT.resolve()),
        "bundle_layout_complete": all(
            (ROOT / relative).exists()
            for relative in (
                "src/hunyuanocr_demo.py",
                "scripts/start_service.sh",
                "assets/images",
                "outputs",
            )
        ),
        "model_dir": str(MODEL_DIR),
        "cache_dir": os.environ.get("VLLM_CACHE_ROOT"),
        "output_dir": str(OUTPUT_DIR),
        "workspace_exists": Path("/hunyuanOCR_workspace").is_dir(),
        "model_weight_exists": (MODEL_DIR / "model.safetensors").is_file(),
        "model_weight_bytes": (MODEL_DIR / "model.safetensors").stat().st_size
        if (MODEL_DIR / "model.safetensors").is_file() else None,
        "output_writable": os.access(OUTPUT_DIR, os.W_OK),
    }
    report = {
        "base_image_digest": os.environ.get("BASE_IMAGE_DIGEST", BASE_IMAGE_DIGEST),
        "notebook_image": os.environ.get("NOTEBOOK_IMAGE_NAME"),
        "os": os_release,
        "python": os.sys.version.split()[0],
        "packages": packages,
        "paths": paths,
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocm_smi": _command(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"], 20),
    }
    if include_torch_probe:
        code = (
            "import json,torch; "
            "print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
            "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),"
            "'devices':[{'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory,"
            "'arch':getattr(torch.cuda.get_device_properties(i),'gcnArchName',None)} for i in range(torch.cuda.device_count())]}))"
        )
        probe = _command(["/opt/venv/bin/python3", "-c", code], 120)
        if probe["returncode"] == 0 and probe["stdout"]:
            probe["json"] = json.loads(probe["stdout"].splitlines()[-1])
        report["torch_probe"] = probe
    return report


def inspect_runtime() -> dict[str, Any]:
    model_files = []
    if MODEL_DIR.is_dir():
        for path in sorted(MODEL_DIR.rglob("*")):
            if path.is_file():
                model_files.append({
                    "path": str(path.relative_to(MODEL_DIR)),
                    "bytes": path.stat().st_size,
                })
    modules = {}
    for name in ("openai", "PIL", "torch", "transformers", "vllm"):
        try:
            __import__(name)
            modules[name] = True
        except Exception as error:
            modules[name] = f"{type(error).__name__}: {error}"
    return {
        "workspace": str(Path("/hunyuanOCR_workspace")),
        "entrypoint": "/hunyuanOCR_workspace/bin/entrypoint.sh",
        "vllm_binary": shutil.which("vllm"),
        "model_dir": str(MODEL_DIR),
        "model_files": model_files,
        "model_file_count": len(model_files),
        "modules": modules,
        "configuration": {
            "model": MODEL_NAME,
            "tensor_parallel": 1,
            "max_model_len": 131072,
            "gpu_memory_utilization": 0.90,
            "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
            "attention_backend": os.environ.get(
                "VLLM_ATTENTION_BACKEND", "ROCM_ATTN"
            ),
        },
    }


def service_status(timeout: float = 2.0) -> dict[str, Any]:
    pid_file = RUNTIME_DIR / "vllm.pid"
    backend_file = RUNTIME_DIR / "attention_backend"
    try:
        pid = int(pid_file.read_text().strip()) if pid_file.is_file() else None
    except (OSError, ValueError):
        pid = None
    pid_state = None
    if pid is not None:
        try:
            pid_state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (OSError, IndexError):
            pass
    pid_alive = pid_state is not None and pid_state != "Z"
    try:
        models = _http_json(f"{base_url()}/models", timeout)
        ready = any(item.get("id") == MODEL_NAME for item in models.get("data", []))
        error = None
    except Exception as exc:
        models = None
        ready = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "ready": ready,
        "pid": pid,
        "pid_alive": pid_alive,
        "pid_state": pid_state,
        "attention_backend": backend_file.read_text().strip()
        if backend_file.is_file()
        else None,
        "models": models,
        "error": error,
    }


def start_service(timeout_seconds: int = 1800) -> dict[str, Any]:
    ensure_layout()
    desired_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "ROCM_ATTN")
    before = service_status()
    if before["ready"]:
        active_backend = before["attention_backend"]
        if active_backend != desired_backend:
            raise RuntimeError(
                f"vLLM already uses {active_backend or 'an unknown backend'}; "
                f"stop_service() before selecting {desired_backend}."
            )
        return {"started": False, "elapsed_seconds": 0.0, **before}
    started = time.monotonic()
    env = os.environ.copy()
    env["STARTUP_TIMEOUT"] = str(timeout_seconds)
    completed = subprocess.run(
        [str(ROOT / "scripts" / "start_service.sh")],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds + 60,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"vLLM startup failed after {elapsed:.1f}s\n{completed.stdout}\n{completed.stderr}\n"
            f"Log tail:\n{tail_service_log()}"
        )
    status = service_status(10)
    if not status["ready"]:
        raise RuntimeError(f"start script returned but API is not ready: {status}")
    return {
        "started": True,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        **status,
    }


def stop_service() -> dict[str, Any]:
    completed = subprocess.run(
        [str(ROOT / "scripts" / "stop_service.sh")],
        text=True,
        capture_output=True,
        timeout=120,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        **service_status(),
    }


def tail_service_log(lines: int = 80) -> str:
    path = LOG_DIR / "vllm.log"
    if not path.is_file():
        return "vLLM log does not exist yet"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def list_images(directory: Path = ASSETS_DIR) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _normalized_image(source: str | Path | Image.Image, max_side: int = 4096):
    if isinstance(source, Image.Image):
        image = source.copy()
        source_path = None
    else:
        source_path = Path(source).resolve()
        with Image.open(source_path) as opened:
            image = opened.copy()
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image, source_path, original_size


def _data_url(image: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def classify_output(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    return {
        "empty": not stripped,
        "coordinate_only": bool(stripped and COORDINATE_ONLY.fullmatch(stripped)),
        "too_short": bool(stripped and len(stripped) < 20),
        "characters": len(stripped),
        "passed": bool(stripped) and not COORDINATE_ONLY.fullmatch(stripped) and len(stripped) >= 20,
    }


def _markdown_blocks(text: str) -> list[dict[str, Any]]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text.strip()) if chunk.strip()]
    blocks = []
    for index, chunk in enumerate(chunks, 1):
        lowered = chunk.lower()
        if "<table" in lowered:
            block_type = "table_html"
        elif chunk.startswith("#"):
            block_type = "heading"
        elif "\\[" in chunk or "$$" in chunk:
            block_type = "formula_or_text"
        else:
            block_type = "text"
        blocks.append({
            "index": index,
            "type": block_type,
            "text": chunk,
            "bbox": None,
            "confidence": None,
        })
    return blocks


def infer_image(
    source: str | Path | Image.Image,
    prompt: str = DOC_PARSE_PROMPT,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.08,
) -> dict[str, Any]:
    start_service(timeout_seconds=1800)
    image, source_path, original_size = _normalized_image(source)
    encoded = _data_url(image)
    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url(),
        timeout=3600.0,
        http_client=httpx.Client(trust_env=False, timeout=3600.0),
    )
    started = time.monotonic()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encoded}},
                {"type": "text", "text": prompt},
            ]},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body={
            "top_k": -1,
            "repetition_penalty": repetition_penalty,
            "skip_special_tokens": True,
        },
    )
    elapsed = time.monotonic() - started
    text = response.choices[0].message.content or ""
    usage = response.usage
    source_bytes = source_path.read_bytes() if source_path else image.tobytes()
    result = {
        "schema_version": "1.0",
        "status": "PASS" if classify_output(text)["passed"] else "QUALITY_WARNING",
        "model": MODEL_NAME,
        "task": "doc_parse",
        "prompt": prompt,
        "source": {
            "path": str(source_path) if source_path else None,
            "name": source_path.name if source_path else "PIL.Image",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "original_size": list(original_size),
            "submitted_size": list(image.size),
            "megapixels": round(image.width * image.height / 1_000_000, 4),
        },
        "text": text,
        "blocks": _markdown_blocks(text),
        "timing": {"latency_seconds": round(elapsed, 6)},
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "finish_reason": response.choices[0].finish_reason,
        "quality": classify_output(text),
        "spatial_fields": {
            "available": False,
            "note": "doc_parse returns reading-order Markdown; it does not expose calibrated boxes or confidence scores.",
        },
    }
    return result


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("_") or "result"


def save_result(result: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    stem = _safe_stem(result["source"]["name"])
    json_dir = output_dir / "json"
    text_dir = output_dir / "text"
    json_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{stem}.json"
    markdown_path = text_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(result["text"])
    return {"json": json_path, "markdown": markdown_path}


def batch_infer(
    directory: Path = ASSETS_DIR,
    output_dir: Path = OUTPUT_DIR,
    resume: bool = True,
) -> list[dict[str, Any]]:
    results = []
    images = list_images(directory)
    for index, path in enumerate(images, 1):
        json_path = output_dir / "json" / f"{_safe_stem(path.name)}.json"
        if resume and json_path.is_file():
            result = json.loads(json_path.read_text())
            action = "resume"
        else:
            try:
                result = infer_image(path)
                save_result(result, output_dir)
                action = "infer"
            except Exception as error:
                result = {
                    "schema_version": "1.0",
                    "status": "ERROR",
                    "source": {"path": str(path), "name": path.name},
                    "text": "",
                    "blocks": [],
                    "timing": {"latency_seconds": None},
                    "quality": {"passed": False},
                    "error": f"{type(error).__name__}: {error}",
                }
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                action = "error"
        print(f"[{index}/{len(images)}] {action:6s} {path.name}")
        results.append(result)
    return results


def summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        source = result.get("source", {})
        rows.append({
            "file": source.get("name"),
            "size": "x".join(map(str, source.get("submitted_size", []))),
            "megapixels": source.get("megapixels"),
            "characters": len(result.get("text", "")),
            "blocks": len(result.get("blocks", [])),
            "completion_tokens": result.get("usage", {}).get("completion_tokens"),
            "latency_seconds": result.get("timing", {}).get("latency_seconds"),
            "quality_pass": result.get("quality", {}).get("passed", False),
            "status": result.get("status"),
        })
    return rows


def write_markdown_report(results: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or OUTPUT_DIR / "hunyuanocr_report.md"
    lines = ["# HunyuanOCR 1.5 Demo Report", "", f"Model: `{MODEL_NAME}`", ""]
    for result in results:
        name = result.get("source", {}).get("name", "unknown")
        latency = result.get("timing", {}).get("latency_seconds")
        lines.extend([
            f"## {name}", "",
            f"- Status: `{result.get('status')}`",
            f"- Latency: `{latency}` seconds",
            f"- Quality: `{result.get('quality', {})}`",
            "", result.get("text", ""), "",
        ])
    path.write_text("\n".join(lines))
    return path


def archive_outputs(path: Path | None = None) -> Path:
    path = path or ROOT / "hunyuanocr_demo_outputs.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(OUTPUT_DIR.rglob("*")):
            if item.is_file() and item != path:
                archive.write(item, item.relative_to(ROOT))
    return path


def performance_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        row["latency_seconds"] for row in summary_rows(results)
        if isinstance(row["latency_seconds"], (int, float))
    ]
    return {
        "samples": len(latencies),
        "mean_seconds": round(statistics.mean(latencies), 6) if latencies else None,
        "stdev_seconds": round(statistics.stdev(latencies), 6) if len(latencies) > 1 else 0.0,
        "images_per_minute": round(60 / statistics.mean(latencies), 4) if latencies else None,
    }


def diagnostic_report(path: Path | None = None) -> dict[str, Any]:
    path = path or OUTPUT_DIR / "diagnostics.json"
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": runtime_report(include_torch_probe=False),
        "service": service_status(),
        "inspection": inspect_runtime(),
        "service_log_tail": tail_service_log(),
        "common_failures": {
            "cuda_oom": "Stop other GPU jobs, then restart this container and the vLLM service.",
            "font_missing": "Rebuild the image; fonts-noto-cjk is installed by Dockerfile.",
            "corrupt_image": "Pillow must be able to load and convert the file to RGB.",
            "model_path": f"Expected model path: {MODEL_DIR}",
            "import_failure": "Run scripts/inspect_runtime.py and attach outputs/diagnostics.json.",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
