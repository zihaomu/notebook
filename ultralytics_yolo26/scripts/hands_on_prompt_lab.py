#!/usr/bin/env python3
"""Prompt-focused Hands-on helpers for fixed YOLO/VLM video evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2

from scripts import notebook_env as env

DEFAULT_QUESTION = "What is happening in this scene?"
ROLE = "You are a careful visual observer."
FORMAT = "Answer in one sentence of no more than 20 words."
GROUNDING = (
    "Use only visible evidence. Do not invent unclear details; "
    "say unclear when the evidence is insufficient."
)


def compose_prompt(question: str) -> str:
    question = " ".join(str(question).strip().split())
    if not question:
        raise ValueError("MY_QUESTION cannot be empty")
    return " ".join((ROLE, question, FORMAT, GROUNDING))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frames


@dataclass
class PromptLab:
    output_dir: Path
    baseline_dir: Path
    source_video: Path
    yolo_video: Path
    baseline_video: Path
    baseline_timeline_path: Path
    client: object

    @classmethod
    def prepare(cls, output_dir: Path | None = None, client=None):
        from scripts import pipeline_workflow as workflow
        from vlm_client import LlamaCppVLMClient

        workflow = importlib.reload(workflow)
        output_dir = Path(output_dir or env.OUTPUT)
        output_dir.mkdir(parents=True, exist_ok=True)
        workflow.run_workflow(force=False)
        if client is None:
            client = LlamaCppVLMClient(base_url=env.LLAMACPP_BASE_URL)
            if not client.health_check():
                raise RuntimeError(f"VLM unavailable: {client.base_url}")
        return cls(
            output_dir=output_dir,
            baseline_dir=workflow.RUN_DIR,
            source_video=env.SOURCE_VIDEO,
            yolo_video=workflow.YOLO_VIDEO,
            baseline_video=workflow.FINAL_VIDEO,
            baseline_timeline_path=workflow.TIMELINE,
            client=client,
        )

    @property
    def baseline_timeline(self) -> dict:
        return json.loads(self.baseline_timeline_path.read_text(encoding="utf-8"))

    def evidence(self, segment_index: int = 0) -> dict:
        segment = self.baseline_timeline["segments"][segment_index]
        storyboard = self.baseline_dir / segment["storyboard"]
        if not storyboard.is_file():
            raise FileNotFoundError(storyboard)
        return {
            "segment_index": segment_index,
            "start": segment["start"],
            "end": segment["end"],
            "sample_times": segment["sample_times"],
            "storyboard": str(storyboard),
            "sha256": sha256(storyboard),
            "baseline_caption": segment["caption"],
            "baseline_latency_seconds": segment["latency_seconds"],
        }

    def compare(self, question: str, segment_index: int = 0) -> dict:
        evidence = self.evidence(segment_index)
        image = cv2.imread(evidence["storyboard"])
        if image is None:
            raise RuntimeError(f"Cannot read storyboard: {evidence['storyboard']}")
        custom_prompt = compose_prompt(question)
        started = time.perf_counter()
        custom_answer = self.client.describe_roi(image, custom_prompt)
        custom_latency = time.perf_counter() - started
        result = {
            "evidence": evidence,
            "default": {
                "question": DEFAULT_QUESTION,
                "prompt": self.baseline_timeline["prompt"],
                "answer": evidence["baseline_caption"],
                "latency_seconds": evidence["baseline_latency_seconds"],
            },
            "custom": {
                "question": " ".join(question.strip().split()),
                "prompt": custom_prompt,
                "answer": " ".join(custom_answer.strip().split()),
                "latency_seconds": round(custom_latency, 3),
            },
        }
        (self.output_dir / "prompt_comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

    def run_timeline(self, question: str, comparison: dict | None = None,
                     force: bool = False) -> dict:
        baseline = self.baseline_timeline
        normalized_question = " ".join(question.strip().split())
        prompt = compose_prompt(normalized_question)
        timeline_path = self.output_dir / "custom_timeline.json"
        video_path = self.output_dir / "custom_prompt_video.mp4"
        if not force and timeline_path.is_file() and video_path.is_file():
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            if timeline.get("prompt") == prompt:
                self._validate_video(video_path)
                return timeline

        segments = []
        for segment_index, segment in enumerate(baseline["segments"]):
            storyboard = self.baseline_dir / segment["storyboard"]
            evidence_sha = sha256(storyboard)
            can_reuse = (
                segment_index == 0
                and comparison is not None
                and comparison["custom"]["question"] == normalized_question
                and comparison["evidence"]["sha256"] == evidence_sha
            )
            if can_reuse:
                answer = comparison["custom"]["answer"]
                latency = comparison["custom"]["latency_seconds"]
            else:
                image = cv2.imread(str(storyboard))
                if image is None:
                    raise RuntimeError(f"Cannot read storyboard: {storyboard}")
                started = time.perf_counter()
                answer = self.client.describe_roi(image, prompt)
                latency = time.perf_counter() - started
            segments.append({
                **segment,
                "caption": " ".join(answer.strip().split()),
                "raw_response": answer,
                "latency_seconds": latency,
                "evidence_sha256": evidence_sha,
            })

        timeline = {
            **baseline,
            "prompt": prompt,
            "question": normalized_question,
            "segments": segments,
        }
        timeline_path.write_text(
            json.dumps(timeline, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._render(timeline, video_path)
        return timeline

    def video_paths(self) -> dict[str, Path]:
        custom = self.output_dir / "custom_prompt_video.mp4"
        self._validate_video(self.baseline_video)
        self._validate_video(custom)
        return {"baseline": self.baseline_video, "custom": custom}

    def save_submission(self, question: str, comparison: dict, timeline: dict,
                        conclusion: str) -> Path:
        conclusion = " ".join(conclusion.strip().split())
        if not conclusion:
            raise ValueError("conclusion cannot be empty")
        custom_answer = comparison["custom"]["answer"]
        payload = {
            "question": " ".join(question.strip().split()),
            "evidence_sha256": comparison["evidence"]["sha256"],
            "default_question": comparison["default"]["question"],
            "default_answer": comparison["default"]["answer"],
            "custom_answer": custom_answer,
            "segment_answers": [
                {
                    "time": f"{segment['start']:.2f}-{segment['end']:.2f}s",
                    "evidence_sha256": segment["evidence_sha256"],
                    "answer": segment["caption"],
                }
                for segment in timeline["segments"]
            ],
            "automatic_checks": {
                "focus_changed": comparison["default"]["answer"] != custom_answer,
                "format_within_20_words": len(custom_answer.split()) <= 20,
            },
            "participant_conclusion": conclusion,
        }
        output = self.output_dir / "submission.json"
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    def _render(self, timeline: dict, output: Path) -> None:
        from scene_subtitle_pipeline import DEFAULT_FONT, render

        args = SimpleNamespace(
            annotated_input=str(self.yolo_video),
            output=str(output),
            font=DEFAULT_FONT,
            band_height=140,
            video_encode="vaapi",
        )
        render(args, timeline)
        self._validate_video(output)

    def _validate_video(self, path: Path) -> None:
        expected = video_frame_count(self.source_video)
        actual = video_frame_count(path)
        if actual != expected:
            raise RuntimeError(f"Video frame count is {actual}, expected {expected}: {path}")
