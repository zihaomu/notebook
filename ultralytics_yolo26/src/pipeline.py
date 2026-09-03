#!/usr/bin/env python3
"""
Ultralytics YOLO26x Production Video Pipeline on AMD Radeon

Pipeline:
  Video Input → OpenCV HIP Preprocess → Ultralytics YOLO26x (MIGraphX EP)
  → NMS + ROI Crop → Qwen3-VL Q8_0 (llama.cpp) → Overlay + Output
"""

import argparse
import json
import os
import sys
import time

# Ensure the HIP-enabled OpenCV and ROCm installs are found first.
_opencv_install = os.environ.get("OPENCV_INSTALL", "/opt/opencv5")
for candidate in (
    f"{_opencv_install}/lib/python3.10/site-packages",
    f"{_opencv_install}/lib/python3.12/site-packages",
    f"{_opencv_install}/lib/python3.12/dist-packages",
    "/opencv_workspace/install/lib/python3.12/dist-packages",
    "/opt/rocm/lib",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

# Default to a working GPU if not overridden by HIP_VISIBLE_DEVICES
if "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np

import config
from preprocess import GPUPreprocessor, preprocess_frame, preprocess_frame_cpu
from detector import UltralyticsYOLODetector
from vlm_client import AsyncVLMClient, validate_llamacpp_service
from postprocess import draw_detections, draw_scene_panel, draw_stats
from video_io import make_gpu_writer, make_reader, make_writer


def check_gpu():
    """Check GPU availability and print info."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            print(f"[pipeline] AMD GPU via HIP: {count} device(s) available")
            cv2.cuda.setDevice(config.GPU_DEVICE_ID)
            cv2.cuda.printCudaDeviceInfo(config.GPU_DEVICE_ID)
            return True
        else:
            print("[pipeline] No GPU devices found, falling back to CPU preprocessing")
            return False
    except Exception as e:
        print(f"[pipeline] GPU check failed ({e}), falling back to CPU preprocessing")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO26x + OpenCV HIP video pipeline on AMD Radeon"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input video file, RTSP URL, or camera index (e.g., 0)",
    )
    parser.add_argument(
        "--output", "-o", default="output.mp4",
        help="Output video file path (default: output.mp4)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Display output in window (requires X11)",
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Skip VLM stage for faster processing",
    )
    parser.add_argument(
        "--vlm-url", default=None,
        help="Override VLM server base URL (e.g. http://localhost:8199/v1)",
    )
    parser.add_argument(
        "--vlm-model", default=None,
        help="Override VLM model name (llama.cpp: use 'auto' to detect)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames to process (0 = all)",
    )
    parser.add_argument(
        "--vlm-interval", type=int, default=30,
        help="Run VLM every N frames (default: 30)",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--video-decode", default="auto", choices=["auto", "rocdecode", "cpu"],
        help="Video decode backend: 'rocdecode' (GPU/VCN), 'cpu' (OpenCV), or 'auto'",
    )
    parser.add_argument(
        "--video-encode", default="auto", choices=["auto", "vaapi", "cpu"],
        help="Video encode backend: 'vaapi' (GPU/VCN), 'cpu' (OpenCV), or 'auto'",
    )
    parser.add_argument(
        "--gpu-direct-encode",
        default=os.environ.get("GPU_DIRECT_ENCODE", "auto"),
        choices=["auto", "on", "off"],
        help="HIP overlay plus DRM PRIME VAAPI encode: auto, on (required), or off",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config.GPU_DEVICE_ID = args.device

    print("=" * 70)
    print("  Ultralytics YOLO26x Production Video Pipeline on AMD Radeon")
    _vlm_label = "no VLM" if args.no_vlm else "Qwen3-VL Q8_0 + llama.cpp"
    print(f"  Ultralytics ONNX + MIGraphX EP | OpenCV HIP video | {_vlm_label}")
    print("=" * 70)
    print()

    # --- Check GPU ---
    has_gpu = check_gpu()

    # --- Resolve input source ---
    try:
        src = int(args.input)
        is_file = False
    except ValueError:
        src = args.input
        is_file = True

    # Read metadata via a short-lived OpenCV handle (fps / size / frame count).
    _meta = cv2.VideoCapture(src)
    if not _meta.isOpened():
        print(f"[pipeline] ERROR: cannot open input: {args.input}")
        sys.exit(1)
    fps_in = _meta.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    _meta.release()

    # --- Open video input (GPU rocDecode when available/requested) ---
    want_gpu_decode = has_gpu and is_file and args.video_decode in ("auto", "rocdecode")
    reader, dec_kind = make_reader(src, device_id=args.device, prefer_gpu=want_gpu_decode)
    if args.video_decode == "rocdecode" and dec_kind != "rocdecode":
        print("[pipeline] ERROR: rocDecode requested but unavailable")
        sys.exit(1)
    print(f"[pipeline] Input: {args.input} ({width}x{height} @ {fps_in:.1f}fps, "
          f"{total_frames} frames) | decode: {dec_kind}")

    # --- Init video writer (HIP/DRM PRIME direct path when compatible) ---
    gpu_decode = dec_kind == "rocdecode"
    want_gpu_encode = args.video_encode in ("auto", "vaapi")
    direct_eligible = (
        gpu_decode and has_gpu and args.no_vlm and not args.display and want_gpu_encode
    )
    if args.gpu_direct_encode == "on" and not direct_eligible:
        raise RuntimeError(
            "GPU direct encode requires rocDecode, --no-vlm, no display, and VA-API"
        )
    writer = None
    enc_kind = None
    if direct_eligible and args.gpu_direct_encode != "off":
        try:
            writer, enc_kind = make_gpu_writer(
                args.output, width, height, fps_in,
                device=os.environ.get("VAAPI_DEVICE"),
            )
        except Exception as error:
            if args.gpu_direct_encode == "on":
                raise
            print(
                f"[pipeline] GPU direct encode unavailable ({error}); "
                "using host-frame writer"
            )
    if writer is None:
        writer, enc_kind = make_writer(
            args.output, width, height, fps_in, prefer_gpu=want_gpu_encode,
            device=os.environ.get("VAAPI_DEVICE"),
        )
    if args.video_encode == "vaapi" and enc_kind not in ("vaapi", "vaapi-direct"):
        print("[pipeline] ERROR: VA-API encode requested but unavailable")
        sys.exit(1)
    gpu_direct = enc_kind == "vaapi-direct"
    print(f"[pipeline] Output: {args.output} | encode: {enc_kind}")

    # --- Init YOLO detector ---
    print()
    detector = UltralyticsYOLODetector(device_id=args.device)
    print(json.dumps(detector.provider_info(), indent=2))

    # --- Init VLM client (async — never blocks the main loop) ---
    vlm = None
    if not args.no_vlm:
        config.validate_model_files()
        service = validate_llamacpp_service(args.vlm_url)
        print(json.dumps(service, indent=2))
        vlm = AsyncVLMClient(
            backend="llamacpp",
            base_url=args.vlm_url,
            model_name=args.vlm_model,
        )
        if vlm.health_check():
            print(f"[pipeline] VLM connected (async, {vlm.backend_name}): {vlm.base_url}")
        else:
            print(f"[pipeline] WARNING: VLM not available at {vlm.base_url}, running without VLM")
            vlm = None

    print()
    print("[pipeline] Starting pipeline...")
    print("-" * 70)

    preprocess_fn = preprocess_frame if has_gpu else preprocess_frame_cpu
    gpu_preprocessor = (
        GPUPreprocessor(config.INPUT_SIZE, device=f"cuda:{args.device}")
        if gpu_decode else None
    )

    # --- Timing accumulators ---
    t_preprocess = 0.0
    t_detect = 0.0
    t_vlm = 0.0
    t_postprocess = 0.0
    t_frame_download = 0.0
    t_encode = 0.0
    frame_count = 0
    vlm_descriptions = []  # persist VLM results across frames

    t_start = time.time()

    while True:
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
        if gpu_decode:
            ret, rgb_gpu = reader.read_gpu()   # (H, W, 3) uint8 RGB on GPU
        else:
            ret, frame = reader.read()
        if not ret:
            break

        frame_count += 1

        # --- Stage 1: Preprocessing + Stage 2: Detection ---
        t0 = time.time()
        if gpu_decode:
            blob_gpu, scale, pad_w, pad_h = gpu_preprocessor.process(rgb_gpu)
            orig_shape = (int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1]), 3)
            t1 = time.time()
            t_preprocess += t1 - t0
            detections = detector.detect_and_parse_gpu(
                blob_gpu, scale, pad_w, pad_h, orig_shape
            )
            if not gpu_direct:
                download_started = time.time()
                frame = cv2.cvtColor(rgb_gpu.cpu().numpy(), cv2.COLOR_RGB2BGR)
                t_frame_download += time.time() - download_started
        else:
            blob, scale, pad_w, pad_h = preprocess_fn(frame)
            t1 = time.time()
            t_preprocess += t1 - t0
            detections = detector.detect_and_parse(
                blob, scale, pad_w, pad_h, frame.shape
            )
        t2 = time.time()
        t_detect += t2 - t1

        if not gpu_direct:
            if vlm and detections and (
                frame_count % args.vlm_interval == 0 or frame_count == 1
            ):
                if getattr(vlm, "is_gpu_ipc", False) and gpu_decode:
                    vlm.submit_rois_gpu(rgb_gpu.clone(), detections)
                else:
                    vlm.submit_rois(frame.copy(), detections)
            if vlm:
                vlm_descriptions = vlm.get_latest()
            t3 = time.time()
            t_vlm += t3 - t2

            draw_detections(
                frame,
                detections,
                vlm_descriptions if vlm_descriptions else None,
                names=detector.names,
            )
            if vlm_descriptions:
                draw_scene_panel(frame, vlm_descriptions, names=detector.names)

            elapsed = time.time() - t_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            stats = {
                "FPS": f"{current_fps:.1f}",
                "Frame": f"{frame_count}/{total_frames}",
                "Detections": str(len(detections)),
                "Preprocess": f"{(t1-t0)*1000:.1f}ms",
                "Detect": f"{(t2-t1)*1000:.1f}ms",
            }
            if vlm:
                latency = vlm.last_latency
                stats["VLM"] = f"{latency*1000:.0f}ms" if latency > 0 else "pending"
            draw_stats(frame, stats)
            t4 = time.time()
            t_postprocess += t4 - t3
            encode_started = time.time()
            writer.write(frame)
            t_encode += time.time() - encode_started
        else:
            elapsed = time.time() - t_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            status_lines = (
                f"FPS {current_fps:.1f}",
                f"FRAME {frame_count}/{total_frames}",
                f"DETECTIONS {len(detections)}",
                f"PREPROCESS {(t1-t0)*1000:.1f}MS",
                f"DETECT {(t2-t1)*1000:.1f}MS",
            )
            encode_started = time.time()
            writer.write_gpu(
                rgb_gpu, detections, detector.names, status_lines=status_lines
            )
            t_encode += time.time() - encode_started

        if args.display:
            cv2.imshow("AMD Radeon Vision AI Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Dets: {len(detections)}")

    # --- Cleanup ---
    reader.release()
    writer.release()
    direct_writer_info = writer.info() if gpu_direct else None
    if args.display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start

    print()
    print("=" * 70)
    print(f"  Pipeline Complete")
    print(f"  Frames: {frame_count} | Total: {total_time:.2f}s | Avg FPS: {frame_count/total_time:.1f}")
    print(
        "  Host load average: "
        + ", ".join(f"{value:.2f}" for value in os.getloadavg())
        + f" | logical CPUs: {os.cpu_count()}"
    )
    print(f"  Video decode:    {dec_kind}   |   Video encode: {enc_kind}")
    print(f"  Avg Preprocess:  {t_preprocess/frame_count*1000:.2f}ms/frame {'(GPU/HIP)' if has_gpu else '(CPU)'}")
    print(f"  Avg Detection:   {t_detect/frame_count*1000:.2f}ms/frame ({detector.backend})")
    if vlm:
        vlm_calls = frame_count // args.vlm_interval + 1
        print(f"  Avg VLM:         {t_vlm/vlm_calls*1000:.0f}ms/call ({vlm.backend_name}, every {args.vlm_interval} frames)")
    print(f"  Avg Frame D2H:   {t_frame_download/frame_count*1000:.2f}ms/frame")
    if gpu_direct:
        worker_ms = direct_writer_info["worker_seconds"] / frame_count * 1000
        print("  Avg CPU overlay: 0.00ms/frame (HIP overlay active)")
        print(f"  Avg Direct encode feed: {t_encode/frame_count*1000:.2f}ms/frame")
        print(f"  Avg GPU overlay + direct encode worker: {worker_ms:.2f}ms/frame")
        print(
            "  Direct encode queue: "
            f"depth={direct_writer_info['queue_depth']} | "
            f"submitted={direct_writer_info['submitted_frames']} | "
            f"encoded={direct_writer_info['encoded_frames']}"
        )
    else:
        print(f"  Avg Overlay:     {t_postprocess/frame_count*1000:.2f}ms/frame")
        print(f"  Avg Encode feed: {t_encode/frame_count*1000:.2f}ms/frame")
    print(f"  Output saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
