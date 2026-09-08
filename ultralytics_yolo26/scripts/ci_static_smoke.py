#!/usr/bin/env python3
"""Dependency-free static release checks for the YOLO26 workshop package."""

from __future__ import annotations

import ast
import json
import runpy
import subprocess
from pathlib import Path

from validate_release_lock import parse_lock, validate_values

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_python() -> int:
    count = 0
    for directory in ("scripts", "src", "tests", "docker"):
        for path in sorted((PACKAGE_ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            count += 1
    require(count > 0, "no Python files were checked")
    return count


def check_shell() -> int:
    paths = [
        path
        for directory in ("scripts", "docker")
        for path in sorted((PACKAGE_ROOT / directory).glob("*.sh"))
    ]
    require(paths, "no shell scripts were checked")
    subprocess.run(["bash", "-n", *(str(path) for path in paths)], check=True)
    return len(paths)


def check_notebooks() -> dict[str, int]:
    result: dict[str, int] = {}
    generated = runpy.run_path(str(PACKAGE_ROOT / "scripts/build_notebooks.py"))[
        "NOTEBOOKS"
    ]
    paths = sorted(PACKAGE_ROOT.glob("*.ipynb"))
    require(len(paths) == 2, f"expected two notebooks, found {len(paths)}")
    require(set(generated) == {path.name for path in paths}, "generator output set differs")
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        require(cells, f"{path.name}: no cells")
        ids: list[str] = []
        code_count = 0
        executed_count = 0
        for number, cell in enumerate(cells, 1):
            metadata = cell.get("metadata", {})
            cell_id = metadata.get("id")
            language = metadata.get("language")
            require(bool(cell_id), f"{path.name}: cell {number} has no metadata.id")
            require(bool(language), f"{path.name}: cell {number} has no metadata.language")
            ids.append(cell_id)
            if cell.get("cell_type") == "code":
                code_count += 1
                if cell.get("execution_count") is not None:
                    executed_count += 1
                errors = [
                    output
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                require(not errors, f"{path.name}: cell {number} contains an error output")
        require(len(ids) == len(set(ids)), f"{path.name}: duplicate cell IDs")
        generated_cells = generated[path.name]["cells"]
        require(len(cells) == len(generated_cells), f"{path.name}: generated cell count differs")
        for number, (saved, expected) in enumerate(zip(cells, generated_cells), 1):
            require(saved.get("cell_type") == expected.get("cell_type"), f"{path.name}: cell {number} type differs from generator")
            require(saved.get("source") == expected.get("source"), f"{path.name}: cell {number} source differs from generator")
            for key in ("id", "language"):
                require(saved.get("metadata", {}).get(key) == expected.get("metadata", {}).get(key), f"{path.name}: cell {number} metadata.{key} differs from generator")
        require(code_count > 0, f"{path.name}: no code cells")
        require(
            executed_count == code_count,
            f"{path.name}: only {executed_count}/{code_count} code cells are executed",
        )
        result[path.name] = code_count
    return result


def check_release_and_docs() -> str:
    lock_path = PACKAGE_ROOT / "release/current.env"
    values = parse_lock(lock_path)
    validate_values(values)
    subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "-e", f'{values["WORKSHOP_SOURCE_COMMIT"]}^{{commit}}'],
        check=True,
    )
    archive = PACKAGE_ROOT / "release" / f'{values["WORKSHOP_RELEASE_ID"]}.env'
    require(archive.is_file(), f"missing immutable release archive: {archive.name}")
    require(archive.read_bytes() == lock_path.read_bytes(), "current lock differs from its archive")

    launcher = (PACKAGE_ROOT / "scripts/start_notebook_container.sh").read_text()
    for token in ("release/current.env", "PIPELINE_IMAGE_REF", "LLAMA_IMAGE_REF", "docker pull"):
        require(token in launcher, f"launcher is missing {token}")
    require("zihao/ultralytics-yolo26-workshop" not in launcher, "launcher uses a local pipeline tag")
    require("zihao/llamacpp-q8" not in launcher, "launcher uses a local llama tag")

    build = (PACKAGE_ROOT / "scripts/build_notebook_image.sh").read_text()
    publish = (PACKAGE_ROOT / "scripts/push_notebook_image.sh").read_text()
    require("Refusing to build from a dirty workshop tree" in build, "build clean-tree guard missing")
    require("Refusing to publish from a dirty workshop tree" in publish, "publish clean-tree guard missing")
    require("APPROVED_REGISTRY_IMAGE" in publish, "exact release-tag approval guard missing")
    dockerignore = (PACKAGE_ROOT / ".dockerignore").read_text()
    dockerfile = (PACKAGE_ROOT / "docker/Dockerfile").read_text()
    for token in ("!doc/**", "release/*", "!release/README.md"):
        require(token in dockerignore, f"Docker context is missing {token}")
    require(
        'COPY release /opt/ultralytics-yolo26/seed/release' not in dockerfile,
        "Dockerfile must not bake mutable release locks",
    )
    bundle_source = (PACKAGE_ROOT / "scripts/workshop_bundle_identity.py").read_text()
    require('    "release",' not in bundle_source, "bundle must exclude mutable release locks")
    require('    "release/README.md",' in bundle_source, "bundle must include release policy")
    for token in (
        "COPY doc /opt/ultralytics-yolo26/seed/doc",
        "COPY release/README.md /opt/ultralytics-yolo26/seed/release/README.md",
        "io.ultralytics.release.id",
        "io.ultralytics.companion.digest",
        "ULTRALYTICS_WORKSHOP_SOURCE_COMMIT=${WORKSHOP_GIT_COMMIT}",
        "ULTRALYTICS_WORKSHOP_RELEASE_ID=${WORKSHOP_RELEASE_ID}",
        "ULTRALYTICS_COMPANION_IMAGE_REF=${COMPANION_IMAGE_REF}",
    ):
        require(token in dockerfile, f"Dockerfile is missing {token}")

    english = (PACKAGE_ROOT / "README.md").read_text()
    chinese = (PACKAGE_ROOT / "README_CN.md").read_text()
    model_docs = (PACKAGE_ROOT / "models/README.md").read_text()
    for name, text in (("README.md", english), ("README_CN.md", chinese)):
        require("release/current.env" in text, f"{name}: release lock is not documented")
    require("first notebook cell prepares and strictly validates" not in english.lower(), "README.md has stale model wording")
    require("第一个 notebook 单元会下载并严格校验" not in chinese.lower(), "README_CN.md has stale model wording")
    require("strict offline verification" in model_docs, "models/README.md lacks baked offline semantics")
    return values["WORKSHOP_RELEASE_ID"]


def check_workflow() -> None:
    workflow = REPOSITORY_ROOT / ".github/workflows/ultralytics-yolo26-smoke.yml"
    require(workflow.is_file(), "YOLO26 smoke workflow is missing")
    text = workflow.read_text(encoding="utf-8")
    for token in ("pull_request:", "workflow_dispatch:", "ubuntu-latest", "self-hosted", "w7900"):
        require(token in text, f"workflow is missing {token}")


def main() -> None:
    result = {
        "python_files": check_python(),
        "shell_files": check_shell(),
        "notebooks": check_notebooks(),
        "release_id": check_release_and_docs(),
    }
    check_workflow()
    result["status"] = "PASS"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
