#!/usr/bin/env python3
"""Focused contract tests for the prompt-focused Hands-on helper."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.hands_on_prompt_lab import PromptLab, compose_prompt, sha256


class FakeClient:
    def __init__(self):
        self.prompts = []

    def describe_roi(self, image, prompt):
        assert image.shape == (24, 40, 3)
        self.prompts.append(prompt)
        return f"answer {len(self.prompts)}"


class FakePromptLab(PromptLab):
    def _render(self, timeline, output):
        output.write_bytes(b"fake-video")

    def _validate_video(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        baseline = root / "baseline"
        output = root / "output"
        baseline.mkdir()
        output.mkdir()
        segments = []
        for index in range(4):
            storyboard = baseline / f"storyboard-{index}.jpg"
            assert cv2.imwrite(str(storyboard), np.full((24, 40, 3), index * 25, np.uint8))
            segments.append({
                "index": index + 1,
                "start": index * 4.0,
                "end": min((index + 1) * 4.0, 15.72),
                "sample_times": [index * 4.0 + 0.6, index * 4.0 + 2.0, index * 4.0 + 3.4],
                "storyboard": storyboard.name,
                "caption": f"baseline {index + 1}",
                "raw_response": f"baseline {index + 1}",
                "latency_seconds": 0.4,
            })
        timeline_path = baseline / "timeline.json"
        timeline_path.write_text(json.dumps({
            "source": "source.mp4",
            "backend": "llamacpp",
            "model": "fake.gguf",
            "base_url": "http://fake/v1",
            "prompt": "baseline prompt",
            "interval_seconds": 4.0,
            "video": {"fps": 25.0, "frame_count": 393, "width": 1920, "height": 1080},
            "segments": segments,
        }))
        baseline_video = baseline / "baseline.mp4"
        baseline_video.write_bytes(b"fake-video")
        client = FakeClient()
        lab = FakePromptLab(
            output_dir=output,
            baseline_dir=baseline,
            source_video=root / "source.mp4",
            yolo_video=root / "yolo.mp4",
            baseline_video=baseline_video,
            baseline_timeline_path=timeline_path,
            client=client,
        )
        question = "What road-safety risks are visible?"
        comparison = lab.compare(question)
        assert comparison["evidence"]["sha256"] == sha256(baseline / "storyboard-0.jpg")
        assert comparison["default"]["answer"] == "baseline 1"
        assert comparison["custom"]["answer"] == "answer 1"
        assert client.prompts == [compose_prompt(question)]
        timeline = lab.run_timeline(question, comparison=comparison)
        assert len(timeline["segments"]) == 4
        assert timeline["segments"][0]["caption"] == "answer 1"
        assert [segment["caption"] for segment in timeline["segments"][1:]] == ["answer 2", "answer 3", "answer 4"]
        assert client.prompts == [compose_prompt(question)] * 4
        submission = lab.save_submission(
            question,
            comparison,
            timeline,
            "The question changed the focus while the visual evidence stayed fixed.",
        )
        payload = json.loads(submission.read_text())
        assert payload["evidence_sha256"] == comparison["evidence"]["sha256"]
        assert len(payload["segment_answers"]) == 4
        assert payload["automatic_checks"]["focus_changed"] is True
    print("HANDS_ON_PROMPT_LAB=PASS")


if __name__ == "__main__":
    main()
