#!/usr/bin/env python3
"""Validate asynchronous VLM accounting without a live model service."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from vlm_client import AsyncVLMClient


class FakeVLMClient:
    base_url = "fake://vlm"

    def __init__(self, fail_first: bool = False, drop_last: bool = False) -> None:
        self.fail_first = fail_first
        self.drop_last = drop_last
        self.release = threading.Event()

    def health_check(self) -> bool:
        return True

    def describe_rois(self, frame, detections, top_k):
        self.release.wait(2)
        results = []
        selected = detections[:top_k]
        if self.drop_last:
            selected = selected[:-1]
        for index, detection in enumerate(selected):
            description = "[VLM error: fake]" if self.fail_first and index == 0 else "ok"
            results.append((detection, description))
        return results


def validate_success_and_busy_skip() -> None:
    fake = FakeVLMClient()
    client = AsyncVLMClient(sync_client=fake)
    detections = [(0, 0, 1, 1, 0.9, 0)] * 4

    assert client.submit_rois(object(), detections, top_k=3)
    assert not client.submit_rois(object(), detections, top_k=3)
    fake.release.set()
    assert client.wait(timeout=2)

    metrics = client.metrics()
    assert metrics["submitted_batches"] == 1
    assert metrics["skipped_busy"] == 1
    assert metrics["submitted_rois"] == 3
    assert metrics["completed_batches"] == 1
    assert metrics["completed_rois"] == 3
    assert metrics["failed_rois"] == 0
    assert len(metrics["latest_descriptions"]) == 3


def validate_failed_roi_accounting() -> None:
    fake = FakeVLMClient(fail_first=True)
    client = AsyncVLMClient(sync_client=fake)
    detections = [(0, 0, 1, 1, 0.9, 0)] * 3

    assert client.submit_rois(object(), detections, top_k=3)
    fake.release.set()
    assert client.wait(timeout=2)

    metrics = client.metrics()
    assert metrics["completed_rois"] == 2
    assert metrics["failed_rois"] == 1


def validate_missing_roi_accounting() -> None:
    fake = FakeVLMClient(drop_last=True)
    client = AsyncVLMClient(sync_client=fake)
    detections = [(0, 0, 1, 1, 0.9, 0)] * 3

    assert client.submit_rois(object(), detections, top_k=3)
    fake.release.set()
    assert client.wait(timeout=2)

    metrics = client.metrics()
    assert metrics["submitted_rois"] == 3
    assert metrics["completed_rois"] == 2
    assert metrics["failed_rois"] == 1


def main() -> None:
    validate_success_and_busy_skip()
    validate_failed_roi_accounting()
    validate_missing_roi_accounting()
    print("ASYNC_VLM_CLIENT=PASS")


if __name__ == "__main__":
    main()
