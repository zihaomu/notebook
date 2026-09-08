#!/usr/bin/env python3
"""Validate the immutable two-image release lock."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TAG_RE = re.compile(r"^[A-Za-z0-9._:/-]+:[A-Za-z0-9._-]+$")
IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$")
REQUIRED_KEYS = (
    "WORKSHOP_RELEASE_ID",
    "WORKSHOP_SOURCE_COMMIT",
    "WORKSHOP_BUNDLE_SHA256",
    "WORKSHOP_MODEL_SET_SHA256",
    "PIPELINE_IMAGE_TAG",
    "PIPELINE_IMAGE_REF",
    "PIPELINE_AMD64_MANIFEST",
    "LLAMA_IMAGE_TAG",
    "LLAMA_IMAGE_REF",
    "LLAMA_AMD64_MANIFEST",
)


def parse_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{number}: invalid key {key!r}")
        if key in values:
            raise ValueError(f"{path}:{number}: duplicate key {key}")
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"{path}:{number}: empty or whitespace-containing value")
        values[key] = value
    missing = [key for key in REQUIRED_KEYS if key not in values]
    extras = sorted(set(values) - set(REQUIRED_KEYS))
    if missing or extras:
        raise ValueError(f"release lock keys differ: missing={missing}, extras={extras}")
    return values


def image_ref_digest(reference: str) -> str:
    if not IMAGE_REF_RE.fullmatch(reference):
        raise ValueError(f"invalid immutable image reference: {reference}")
    return reference.rsplit("@", 1)[1]


def validate_values(values: dict[str, str]) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", values["WORKSHOP_RELEASE_ID"]):
        raise ValueError("WORKSHOP_RELEASE_ID is not a portable release identifier")
    if not COMMIT_RE.fullmatch(values["WORKSHOP_SOURCE_COMMIT"]):
        raise ValueError("WORKSHOP_SOURCE_COMMIT must be a full lowercase Git SHA")
    for key in (
        "WORKSHOP_BUNDLE_SHA256",
        "WORKSHOP_MODEL_SET_SHA256",
        "PIPELINE_AMD64_MANIFEST",
        "LLAMA_AMD64_MANIFEST",
    ):
        if not DIGEST_RE.fullmatch(f"sha256:{values[key]}" if key.startswith("WORKSHOP_") else values[key]):
            raise ValueError(f"{key} must be a sha256 digest")
    pipeline_digest = image_ref_digest(values["PIPELINE_IMAGE_REF"])
    llama_digest = image_ref_digest(values["LLAMA_IMAGE_REF"])
    if pipeline_digest == llama_digest:
        raise ValueError("pipeline and llama image digests must differ")
    for key in ("PIPELINE_IMAGE_TAG", "LLAMA_IMAGE_TAG"):
        if not TAG_RE.fullmatch(values[key]):
            raise ValueError(f"{key} must be a safe human-readable tagged reference")


def docker_json(*arguments: str) -> object:
    completed = subprocess.run(
        ["docker", *arguments], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def validate_local_images(values: dict[str, str]) -> None:
    pipeline = docker_json("image", "inspect", values["PIPELINE_IMAGE_REF"])[0]
    llama = docker_json("image", "inspect", values["LLAMA_IMAGE_REF"])[0]
    labels = pipeline.get("Config", {}).get("Labels") or {}
    expected_labels = {
        "org.opencontainers.image.revision": values["WORKSHOP_SOURCE_COMMIT"],
        "io.ultralytics.workshop.bundle.sha256": values["WORKSHOP_BUNDLE_SHA256"],
        "io.ultralytics.model-set.sha256": values["WORKSHOP_MODEL_SET_SHA256"],
    }
    mismatches = {
        key: {"expected": expected, "actual": labels.get(key)}
        for key, expected in expected_labels.items()
        if labels.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"pipeline image label mismatch: {mismatches}")
    release_label = labels.get("io.ultralytics.release.id")
    companion_label = labels.get("io.ultralytics.companion.digest")
    is_bootstrap = values["WORKSHOP_RELEASE_ID"].endswith("-bootstrap")
    if not is_bootstrap:
        if release_label != values["WORKSHOP_RELEASE_ID"]:
            raise ValueError(
                f"pipeline release label mismatch: {release_label!r}"
            )
        if companion_label != values["LLAMA_IMAGE_REF"]:
            raise ValueError(
                f"pipeline companion label mismatch: {companion_label!r}"
            )
    if pipeline.get("Architecture") != "amd64" or pipeline.get("Os") != "linux":
        raise ValueError("pipeline image must be linux/amd64")
    if llama.get("Architecture") != "amd64" or llama.get("Os") != "linux":
        raise ValueError("llama image must be linux/amd64")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "test",
            values["LLAMA_IMAGE_REF"],
            "-x",
            "/opt/llama.cpp/build/bin/llama-server",
        ],
        check=True,
    )


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=package_root / "release/current.env")
    parser.add_argument("--check-local-images", action="store_true")
    args = parser.parse_args()
    values = parse_lock(args.lock.resolve())
    validate_values(values)
    if args.check_local_images:
        validate_local_images(values)
    print(
        json.dumps(
            {
                "status": "PASS",
                "release_id": values["WORKSHOP_RELEASE_ID"],
                "source_commit": values["WORKSHOP_SOURCE_COMMIT"],
                "pipeline_image": values["PIPELINE_IMAGE_REF"],
                "llama_image": values["LLAMA_IMAGE_REF"],
                "local_images_checked": args.check_local_images,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
