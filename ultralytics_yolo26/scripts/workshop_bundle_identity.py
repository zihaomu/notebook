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
    "doc/performance_diagnosis_CN.md",
    "ultralytics_yolo26x_step_by_step.ipynb",
    "ultralytics_yolo26x_end_to_end.ipynb",
    "output/.gitkeep",
)
ROOT_DIRECTORIES = (
    "assets",
    "data",
    "docker",
    "native",
    "doc/speedup_blog",
    "output/benchmarks",
    "output/pipeline",
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
        if relative.parts[:2] == ("native", "build"):
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
