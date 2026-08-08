#!/usr/bin/env python3
"""Check bundled YOLO assets and download missing Qwen GGUF files with progress."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts import notebook_env as env

HF_REPO = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 8 * 1024 * 1024


def _in_notebook_kernel() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _format_amount(value: float, unit: str) -> str:
    if unit == "B":
        amount = float(value)
        for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(amount) < 1024.0 or suffix == "TiB":
                return f"{amount:.2f} {suffix}"
            amount /= 1024.0
    if unit == "s":
        return f"{int(value)} s"
    return f"{value:.1f} {unit}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class NotebookProgress:
    """A single fixed Jupyter HTML progress area updated through one display_id."""

    def __init__(
        self,
        *,
        total: int,
        desc: str,
        unit: str,
        initial: int = 0,
        enabled: bool = True,
        leave: bool = True,
        mininterval: float = 1.0,
    ) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.unit = unit
        self.n = int(initial)
        self.initial = int(initial)
        self.enabled = enabled
        self.leave = leave
        self.mininterval = mininterval
        self.started = time.monotonic()
        self.last_refresh = 0.0
        self.handle = None
        self.finished = False
        self.failed_message: str | None = None

    def __enter__(self) -> "NotebookProgress":
        if self.enabled:
            from IPython.display import HTML, display

            self.handle = display(HTML(self._html()), display_id=True)
            self.last_refresh = time.monotonic()
        return self

    def update(self, amount: int) -> None:
        self.n = min(self.total, self.n + int(amount))
        now = time.monotonic()
        if self.enabled and (now - self.last_refresh >= self.mininterval or self.n >= self.total):
            self.refresh(now)

    def refresh(self, now: float | None = None) -> None:
        if not self.enabled or self.handle is None:
            return
        from IPython.display import HTML

        current = now or time.monotonic()
        self.handle.update(HTML(self._html(current)))
        self.last_refresh = current

    def _html(self, now: float | None = None) -> str:
        current = now or time.monotonic()
        elapsed = max(current - self.started, 1e-9)
        transferred = max(self.n, 0)
        session_bytes = max(transferred - self.initial, 0)
        rate = session_bytes / elapsed
        remaining = max(self.total - transferred, 0)
        eta = remaining / rate if rate > 0 else None
        percent = min(100.0, 100.0 * transferred / self.total)
        status = ""
        color = "#0078d4"
        if self.failed_message:
            status = f"Failed: {html.escape(self.failed_message)}"
            color = "#c42b1c"
        elif self.finished:
            status = "Complete" if transferred >= self.total else "Ready"
            color = "#107c10"
        details = (
            f"{_format_amount(transferred, self.unit)} / {_format_amount(self.total, self.unit)}"
            f" &nbsp;·&nbsp; {_format_amount(rate, self.unit)}/s"
            f" &nbsp;·&nbsp; elapsed {_format_duration(elapsed)}"
            f" &nbsp;·&nbsp; ETA {_format_duration(eta)}"
        )
        if status:
            details += f' &nbsp;·&nbsp; <strong style="color:{color}">{status}</strong>'
        return (
            '<div style="border:1px solid #d0d7de;border-radius:6px;padding:10px 12px;'
            'margin:4px 0;background:#fff;font-family:system-ui,sans-serif">'
            '<div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:6px">'
            f'<strong>{html.escape(self.desc)}</strong><span>{percent:5.1f}%</span></div>'
            f'<progress value="{transferred}" max="{self.total}" '
            f'style="width:100%;height:18px;accent-color:{color}"></progress>'
            f'<div style="font-size:12px;color:#57606a;margin-top:5px">{details}</div></div>'
        )

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.finished = exc_type is None
        if exc_value is not None:
            self.failed_message = str(exc_value)
        self.refresh()
        return False


def progress_bar(
    *,
    total: int,
    desc: str,
    unit: str,
    initial: int = 0,
    enabled: bool = True,
    leave: bool = True,
):
    if enabled and _in_notebook_kernel():
        return NotebookProgress(
            total=total,
            desc=desc,
            unit=unit,
            initial=initial,
            enabled=enabled,
            leave=leave,
        )
    options = {
        "total": total,
        "initial": initial,
        "desc": desc,
        "unit": unit,
        "disable": not enabled,
        "leave": leave,
        "mininterval": 1.0,
        "maxinterval": 5.0,
        "dynamic_ncols": True,
    }
    if unit == "B":
        options.update(unit_scale=True, unit_divisor=1024)
    return tqdm(**options)


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    size: int
    sha256: str | None
    source: str
    gguf: bool = False


BUNDLED_MODELS = (
    ModelSpec(
        "yolo26x.onnx",
        223_209_857,
        "97d5165312f2d1957a24d7ae27863a98860ac88656d40fdac8f10de77999bc35",
        "bundled",
    ),
    ModelSpec(
        "yolo26x_compiled.mxr",
        114_408_032,
        "ef790cfc6b403350c4c5281449811e2dc5e7c7e0d546adfa52caf66e36772d9a",
        "bundled",
    ),
)

DOWNLOAD_MODELS = (
    ModelSpec(
        "Qwen3-VL-8B-Instruct-Q8_0.gguf",
        8_709_520_224,
        "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7",
        HF_REPO,
        gguf=True,
    ),
    ModelSpec(
        "mmproj-F16.gguf",
        1_159_030_336,
        "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4",
        HF_REPO,
        gguf=True,
    ),
)
ALL_MODELS = (*BUNDLED_MODELS, *DOWNLOAD_MODELS)


def _basic_status(path: Path, spec: ModelSpec) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        return False, f"wrong size: {actual_size:,} != {spec.size:,}"
    if spec.gguf:
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                return False, "invalid GGUF header"
    return True, "ready"


def model_status(model_dir: Path | None = None) -> list[dict[str, object]]:
    directory = Path(model_dir or env.MODELS).resolve()
    result = []
    for spec in ALL_MODELS:
        path = directory / spec.filename
        ready, detail = _basic_status(path, spec)
        result.append({
            "filename": spec.filename,
            "source": spec.source,
            "expected_gib": round(spec.size / 1024**3, 3),
            "ready": ready,
            "detail": detail,
            "path": str(path),
        })
    return result


def _sha256_prefix(path: Path, progress: bool) -> hashlib._Hash:
    digest = hashlib.sha256()
    total = path.stat().st_size if path.exists() else 0
    with path.open("rb") as stream, progress_bar(
        total=total,
        desc=f"Hashing {path.name}.part",
        unit="B",
        enabled=progress and total > 0,
        leave=False,
    ) as bar:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest


def _download_url(spec: ModelSpec, endpoint: str) -> str:
    repo = quote(spec.source, safe="/")
    filename = quote(spec.filename)
    return f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}"


def download_model(
    spec: ModelSpec,
    model_dir: Path | None = None,
    endpoint: str | None = None,
    progress: bool = True,
) -> Path:
    """Download one model with Range resume, progress, SHA-256, and atomic replace."""
    directory = Path(model_dir or env.MODELS).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.filename
    ready, _ = _basic_status(destination, spec)
    if ready:
        return destination

    partial = directory / f"{spec.filename}.part"
    if partial.exists() and partial.stat().st_size > spec.size:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    digest = _sha256_prefix(partial, progress) if offset else hashlib.sha256()
    if offset == spec.size:
        actual_hash = digest.hexdigest()
        if spec.sha256 and actual_hash != spec.sha256:
            partial.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch for completed partial {spec.filename}: "
                f"{actual_hash} != {spec.sha256}"
            )
        partial.replace(destination)
        return destination
    mirror = endpoint or os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    url = _download_url(spec, mirror)
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        print(
            f"[models] HTTP {response.status_code}; "
            f"resume_offset={offset:,}; total={spec.size:,} bytes"
        )
        if offset and response.status_code != 206:
            offset = 0
            digest = hashlib.sha256()
            partial.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream, progress_bar(
            total=spec.size,
            initial=offset,
            desc=spec.filename,
            unit="B",
            enabled=progress,
        ) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
            stream.flush()
            os.fsync(stream.fileno())

    actual_size = partial.stat().st_size
    if actual_size != spec.size:
        raise RuntimeError(
            f"Incomplete download for {spec.filename}: {actual_size:,} != {spec.size:,} bytes"
        )
    actual_hash = digest.hexdigest()
    if spec.sha256 and actual_hash != spec.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {spec.filename}: {actual_hash} != {spec.sha256}"
        )
    if spec.gguf:
        with partial.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise RuntimeError(f"Invalid GGUF header: {partial}")
    partial.replace(destination)
    return destination


def ensure_models(
    model_dir: Path | None = None,
    endpoint: str | None = None,
    progress: bool = True,
) -> list[dict[str, object]]:
    """Require bundled YOLO files and download missing Qwen files."""
    directory = Path(model_dir or env.MODELS).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    bundled_errors = []
    for spec in BUNDLED_MODELS:
        ready, detail = _basic_status(directory / spec.filename, spec)
        if not ready:
            bundled_errors.append(f"{spec.filename}: {detail}")
    if bundled_errors:
        raise FileNotFoundError(
            "Bundled YOLO model files are missing or invalid:\n  - "
            + "\n  - ".join(bundled_errors)
        )

    missing_bytes = sum(
        max(0, spec.size - (directory / f"{spec.filename}.part").stat().st_size)
        if (directory / f"{spec.filename}.part").exists()
        else spec.size
        for spec in DOWNLOAD_MODELS
        if not _basic_status(directory / spec.filename, spec)[0]
    )
    free_bytes = shutil.disk_usage(directory).free
    if missing_bytes and free_bytes < missing_bytes + 512 * 1024**2:
        raise OSError(
            f"Not enough free space in {directory}: need about "
            f"{missing_bytes / 1024**3:.2f} GiB plus working space, "
            f"have {free_bytes / 1024**3:.2f} GiB"
        )

    for spec in DOWNLOAD_MODELS:
        ready, detail = _basic_status(directory / spec.filename, spec)
        if ready:
            print(f"[models] {spec.filename}: ready")
            continue
        print(f"[models] {spec.filename}: {detail}; downloading from {endpoint or os.environ.get('HF_ENDPOINT', DEFAULT_HF_ENDPOINT)}")
        download_model(spec, directory, endpoint=endpoint, progress=progress)
        print(f"[models] {spec.filename}: download complete")
    return model_status(directory)


def wait_for_llamacpp(
    root_url: str | None = None,
    timeout: int = 300,
    progress: bool = True,
) -> dict[str, object]:
    """Wait for a llama.cpp service that may start after GGUF download completes."""
    base = (root_url or env.LLAMACPP_ROOT_URL).rstrip("/")
    started = time.monotonic()
    with progress_bar(
        total=timeout,
        desc="Waiting for llama.cpp",
        unit="s",
        enabled=progress,
    ) as bar:
        last = 0
        while time.monotonic() - started < timeout:
            elapsed = int(time.monotonic() - started)
            if elapsed > last:
                bar.update(elapsed - last)
                last = elapsed
            try:
                response = requests.get(f"{base}/v1/models", timeout=3)
                if response.ok:
                    payload = response.json()
                    if payload.get("data"):
                        return payload
            except requests.RequestException:
                pass
            time.sleep(1)
    raise TimeoutError(f"llama.cpp did not become ready within {timeout}s: {base}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=env.MODELS)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))
    args = parser.parse_args()
    report = (
        model_status(args.model_dir)
        if args.check_only
        else ensure_models(args.model_dir, args.endpoint, progress=not args.no_progress)
    )
    for item in report:
        print(item)


if __name__ == "__main__":
    main()
