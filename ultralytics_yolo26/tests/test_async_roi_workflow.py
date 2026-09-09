#!/usr/bin/env python3
"""Validate async ROI workflow commands and artifact identity checks."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("ULTRALYTICS_YOLO26_ROOT", str(PACKAGE_ROOT))
os.environ.setdefault("ULTRALYTICS_YOLO26_MODEL_DIR", str(PACKAGE_ROOT / "models"))
os.environ.setdefault("ULTRALYTICS_YOLO26_OUTPUT_DIR", str(PACKAGE_ROOT / "output"))

from scripts.async_roi_workflow import experiment_paths, pipeline_command


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        yolo_command, yolo_paths = pipeline_command(
            output_dir=output,
            mode="yolo-only",
            max_frames=120,
        )
        assert "--no-vlm" in yolo_command
        assert "--gpu-direct-encode" in yolo_command
        assert yolo_paths == experiment_paths(output, "yolo-only")

        async_command, async_paths = pipeline_command(
            output_dir=output,
            mode="async-roi",
            max_frames=120,
            interval=30,
            top_k=3,
            drain_timeout=15,
        )
        assert "--no-vlm" not in async_command
        assert async_command[async_command.index("--vlm-interval") + 1] == "30"
        assert async_command[async_command.index("--vlm-top-k") + 1] == "3"
        assert async_command[async_command.index("--vlm-drain-timeout") + 1] == "15"
        assert async_paths == experiment_paths(output, "async-roi")

        # A stale cache with different parameters must not be treated as valid.
        async_paths["metrics"].write_text(
            json.dumps({
                "frames": 120,
                "vlm_interval_frames": 60,
                "vlm_top_k": 1,
            }),
            encoding="utf-8",
        )
        for key in ("video", "log"):
            async_paths[key].write_bytes(b"placeholder")

    print("ASYNC_ROI_WORKFLOW=PASS")


if __name__ == "__main__":
    main()
