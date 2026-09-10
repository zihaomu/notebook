#!/usr/bin/env python3
"""Compute a stable identity for the immutable workshop bundle."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT_FILES = (
    ".dockerignore",
    ".gitignore",
    "README.md",
    "README_CN.md",
    "ultralytics_yolo26x_step_by_step.ipynb",
    "ultralytics_yolo26x_hands_on.ipynb",
    "ultralytics_yolo26x_end_to_end.ipynb",
    "data/sidewalk.mp4",
    "output/.gitkeep",
    "output/hands_on/custom_prompt_video.mp4",
    "output/hands_on/custom_timeline.json",
    "output/hands_on/prompt_comparison.json",
    "output/hands_on/submission.json",
    "release/README.md",
)
ROOT_DIRECTORIES = (
    "assets",
    "docker",
    "native",
    "doc",
    "output/benchmarks",
    "output/pipeline",
    "output/async_roi_e2e",
    "scripts",
    "src",
    "tests",
)
MODEL_FILES: tuple[str, ...] = ()


def bundle_files(root: Path) -> list[Path]:
    candidates = [root / name for name in (*ROOT_FILES, *MODEL_FILES)]
    for directory in ROOT_DIRECTORIES:
        candidates.extend((root / directory).rglob("*"))
    result = []
    for path in candidates:
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.parts[:2] in {("native", "build"), ("doc", "slide")}:
            continue
        result.append(path)
    return sorted(set(result), key=lambda path: path.relative_to(root).as_posix())


def compute(root: Path) -> str:
    digest = hashlib.sha256()
    for path in bundle_files(root.resolve()):
        relative = path.relative_to(root.resolve()).as_posix()
        digest.update(f"{relative}\0".encode("utf-8"))
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    print(compute(root))


if __name__ == "__main__":
    main()
