#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED_MARKERS = (
    "Verify the Immutable Runtime",
    "Inspect HunyuanOCR Entrypoints",
    "Start the OCR Service",
    "Synthetic Demo Documents",
    "Single-Page OCR",
    "Render Markdown, HTML, and LaTeX",
    "Complex Layout and Table",
    "Batch OCR",
    "Export Results",
    "Performance and GPU Memory",
    "Reproducibility and Diagnostics",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    errors = []
    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    cells = notebook.get("cells") or []
    if len(cells) < 20:
        errors.append(f"expected at least 20 cells, found {len(cells)}")
    ids = []
    text = ""
    for index, cell in enumerate(cells, 1):
        metadata = cell.get("metadata") or {}
        if metadata.get("language") not in {"markdown", "python"}:
            errors.append(f"cell {index}: metadata.language is missing or invalid")
        cell_id = metadata.get("id")
        if not cell_id:
            errors.append(f"cell {index}: metadata.id is required")
        else:
            ids.append(cell_id)
        if cell.get("id") != cell_id:
            errors.append(f"cell {index}: top-level id must match metadata.id")
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
            errors.append(f"cell {index}: notebook must ship unexecuted")
        text += "".join(cell.get("source") or []) + "\n"
    if len(ids) != len(set(ids)):
        errors.append("cell metadata.id values must be unique")
    for marker in REQUIRED_MARKERS:
        if marker not in text:
            errors.append(f"missing section marker: {marker}")
    forbidden = ("/workspace", "pip install", "!pip", "os.system(")
    for marker in forbidden:
        if marker in text:
            errors.append(f"forbidden notebook content: {marker}")
    result = {"status": "PASS" if not errors else "FAIL", "cells": len(cells), "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
