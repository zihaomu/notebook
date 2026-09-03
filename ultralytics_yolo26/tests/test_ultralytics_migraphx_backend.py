#!/usr/bin/env python3
"""Validate Ultralytics ONNX inference with MIGraphX GPU I/O binding."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from ultralytics.nn.backends.onnx import ONNXBackend

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
from migraphx_cache import prepare_cache


def validate(model: Path, iterations: int = 100) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("A visible ROCm GPU is required")

    prepare_cache(model, model.parent / "ort-migraphx-cache", device_id=0)
    backend = ONNXBackend(
        weight=str(model),
        device=torch.device("cuda:0"),
        fp16=True,
    )
    if backend.provider != "MIGraphXExecutionProvider":
        raise RuntimeError(f"Expected MIGraphXExecutionProvider, got {backend.provider}")
    if not backend.use_io_binding:
        raise RuntimeError("MIGraphX GPU I/O binding is disabled")
    if not backend.migraphx_fp16:
        raise RuntimeError("MIGraphX FP16 provider option is disabled")
    session_options = backend.session.get_session_options()
    if session_options.execution_mode.name != "ORT_SEQUENTIAL":
        raise RuntimeError(f"Unexpected ORT execution mode: {session_options.execution_mode}")
    if session_options.intra_op_num_threads != int(
        os.environ.get("ULTRALYTICS_MIGRAPHX_INTRA_OP_THREADS", "1")
    ):
        raise RuntimeError(
            f"Unexpected ORT intra-op threads: {session_options.intra_op_num_threads}"
        )
    if session_options.inter_op_num_threads != 1:
        raise RuntimeError(
            f"Unexpected ORT inter-op threads: {session_options.inter_op_num_threads}"
        )

    input_tensor = torch.zeros(
        (1, 3, 640, 640), dtype=torch.float32, device="cuda"
    )
    input_pointer = input_tensor.data_ptr()
    output = backend(input_tensor)
    if not isinstance(output, list) or len(output) != 1:
        raise RuntimeError(f"Unexpected backend output container: {type(output)}")
    output_tensor = output[0]
    output_pointer = output_tensor.data_ptr()
    if output_tensor.device.type != "cuda":
        raise RuntimeError(f"Output is not GPU-resident: {output_tensor.device}")
    if tuple(output_tensor.shape) != (1, 300, 6):
        raise RuntimeError(f"Unexpected output shape: {tuple(output_tensor.shape)}")
    if not bool(torch.isfinite(output_tensor).all().item()):
        raise RuntimeError("MIGraphX output contains non-finite values")

    input_tensor.fill_(0.5)
    started = time.perf_counter()
    for _ in range(iterations):
        output = backend(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if input_tensor.data_ptr() != input_pointer:
        raise RuntimeError("Input allocation changed during the warm loop")
    if output[0].data_ptr() != output_pointer:
        raise RuntimeError("Output allocation changed during the warm loop")

    return {
        "provider": backend.provider,
        "providers": backend.providers,
        "provider_options": backend.provider_options[backend.provider],
        "io_binding": backend.use_io_binding,
        "migraphx_fp16": backend.migraphx_fp16,
        "ort_execution_mode": session_options.execution_mode.name,
        "ort_intra_op_threads": session_options.intra_op_num_threads,
        "ort_inter_op_threads": session_options.inter_op_num_threads,
        "input_device": str(input_tensor.device),
        "output_device": str(output_tensor.device),
        "input_pointer": input_pointer,
        "output_pointer": output_pointer,
        "iterations": iterations,
        "warm_ms_per_run": round(elapsed * 1000 / iterations, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(validate(args.model, args.iterations), indent=2))


if __name__ == "__main__":
    main()
