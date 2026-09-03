#!/usr/bin/env python3
"""Run or validate the packaged OpenCV + llama.cpp workflow."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import json

from scripts import pipeline_workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate all outputs")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (
        pipeline_workflow.validate_outputs()
        if args.validate_only
        else pipeline_workflow.run_workflow(force=args.force)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
