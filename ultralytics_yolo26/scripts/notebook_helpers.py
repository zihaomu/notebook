"""Small display helpers shared by the pipeline notebooks."""

from __future__ import annotations

import io
import math
import os
import re
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from html import escape
from pathlib import Path
from urllib.parse import quote


_WARNING_PATTERN = re.compile(r"\b(error|failed|failure|unsupported|fallback|mismatch|corrupt|unavailable|out of memory)\b|\[ERR\]", re.IGNORECASE)




def video_info(path: Path) -> dict[str, float | int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fps),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    result["duration_seconds"] = result["frames"] / result["fps"]
    capture.release()
    return result


def frame_at(path: Path, seconds: float):
    import cv2

    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path} at {seconds:.3f}s")
    return image


def show_bgr(image, title: str, size: tuple[int, int] = (14, 8)) -> None:
    import cv2
    import matplotlib.pyplot as plt

    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_bgr_grid(
    images,
    titles,
    *,
    columns: int = 3,
    size: tuple[int, int] | None = None,
) -> None:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    if not images or len(images) != len(titles):
        raise ValueError("images and titles must be non-empty and have equal length")
    columns = min(columns, len(images))
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=size or (6.4 * columns, 4.2 * rows),
        squeeze=False,
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(images):]:
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def show_video(path: Path, title: str, width: int = 960) -> None:
    """Stream a workspace video through Jupyter without embedding Base64 data."""
    from IPython.display import HTML, display

    path = Path(path).absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Video must be inside the notebook directory: {path}") from error
    source = "/files/" + quote(relative)
    display(HTML(
        f'<figure style="margin:0 0 1.25rem 0">'
        f'<figcaption style="font-weight:600;margin-bottom:.45rem">{escape(title)}</figcaption>'
        f'<video controls preload="metadata" width="{int(width)}" '
        f'style="max-width:100%;height:auto;background:#111" src="{source}"></video>'
        f'</figure>'
    ))


@contextmanager
def capture_log(path: Path, label: str, warning_limit: int = 8):
    """Capture Python and native output, surfacing only warning/error lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    python_buffer = io.StringIO()
    caught = None
    native_text = ""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as native_buffer:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(native_buffer.fileno(), 1)
            os.dup2(native_buffer.fileno(), 2)
            with redirect_stdout(python_buffer), redirect_stderr(python_buffer):
                try:
                    yield
                except BaseException as error:
                    caught = error
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            native_buffer.seek(0)
            native_text = native_buffer.read().decode("utf-8", errors="replace")

    text = python_buffer.getvalue() + native_text
    path.write_text(text, encoding="utf-8")
    if caught is not None:
        tail = text.splitlines()[-30:]
        if tail:
            print(f"{label} failed; log tail from {path}:")
            print("\n".join(tail))
        raise caught

    warnings = [
        line.strip() for line in text.splitlines()
        if _WARNING_PATTERN.search(line)
    ]
    print(f"{label}: complete (full log: {path})")
    if warnings:
        print("Warnings:")
        for line in warnings[:warning_limit]:
            print(f"- {line}")
        if len(warnings) > warning_limit:
            print(f"- ... {len(warnings) - warning_limit} more warning lines in the log")
