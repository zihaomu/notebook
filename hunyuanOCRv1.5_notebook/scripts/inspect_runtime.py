#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hunyuanocr_demo import inspect_runtime, runtime_report

print(json.dumps({"runtime": runtime_report(), "inspection": inspect_runtime()}, ensure_ascii=False, indent=2))
