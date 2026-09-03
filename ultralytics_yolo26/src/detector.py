"""Ultralytics YOLO26x ONNX inference with MIGraphX GPU I/O binding."""

import os
import sys

for candidate in (
    "/opt/opencv5/lib/python3.10/site-packages",
    "/opt/opencv5/lib/python3.12/site-packages",
    "/opt/rocm/lib",
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import cv2
import numpy as np

import config
from migraphx_cache import prepare_cache


class UltralyticsYOLODetector:
    """Own a YOLO model and reuse its GPU-resident ONNX backend in a video loop."""

    def __init__(self, model_path=None, device_id=0):
        import torch
        from ultralytics import YOLO

        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.device_id = device_id
        if not torch.cuda.is_available():
            raise RuntimeError("A visible ROCm GPU is required for YOLO inference")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)

        torch.cuda.set_device(device_id)
        self.cache_dir, self.cache_identity = prepare_cache(
            self.model_path, config.MIGRAPHX_CACHE_ROOT, device_id
        )
        print("[detector] Loading YOLO26x ONNX through Ultralytics...")
        print(f"[detector] MIGraphX cache identity: {self.cache_dir.name}")
        self.yolo = YOLO(self.model_path, task="detect")
        warmup = torch.zeros(
            (1, 3, config.INPUT_SIZE[1], config.INPUT_SIZE[0]),
            dtype=torch.float32,
            device=f"cuda:{device_id}",
        )
        self.yolo.predict(
            source=warmup,
            device=device_id,
            half=True,
            imgsz=config.INPUT_SIZE[0],
            batch=1,
            verbose=False,
            save=False,
        )
        self.predictor = self.yolo.predictor
        self._auto_backend = self.predictor.model
        self._onnx_backend = self._auto_backend.backend
        self.backend = "ultralytics-migraphx"
        self.names = self._auto_backend.names

        if self._auto_backend.format != "onnx":
            raise RuntimeError(
                f"Ultralytics selected unexpected format: {self._auto_backend.format}"
            )
        if self._onnx_backend.provider != "MIGraphXExecutionProvider":
            raise RuntimeError(
                "Ultralytics did not select MIGraphXExecutionProvider: "
                f"{self._onnx_backend.providers}"
            )
        if not self._onnx_backend.use_io_binding:
            raise RuntimeError("MIGraphX GPU I/O binding is disabled")
        if not self._onnx_backend.migraphx_fp16:
            raise RuntimeError("MIGraphX FP16 provider option is disabled")

        output = self.infer_gpu(warmup)
        self._input_pointer = warmup.data_ptr()
        self._output_pointer = output.data_ptr()
        print(
            "[detector] Ultralytics ONNX + MIGraphX EP "
            f"(FP16, GPU I/O binding) ready; output={tuple(output.shape)}"
        )

    def provider_info(self):
        return {
            "owner": "ultralytics.YOLO",
            "ultralytics_format": self._auto_backend.format,
            "provider": self._onnx_backend.provider,
            "providers": self._onnx_backend.providers,
            "provider_options": self._onnx_backend.provider_options[
                self._onnx_backend.provider
            ],
            "io_binding": self._onnx_backend.use_io_binding,
            "migraphx_fp16": self._onnx_backend.migraphx_fp16,
            "cache_dir": str(self.cache_dir),
            "cache_identity": self.cache_identity,
            "output_pointer": self._onnx_backend.bindings[0].data_ptr(),
        }

    def infer_gpu(self, input_tensor):
        """Run only the Ultralytics inference layer on a resident ROCm tensor."""
        import torch

        expected = (1, 3, config.INPUT_SIZE[1], config.INPUT_SIZE[0])
        if input_tensor.device.type != "cuda":
            raise ValueError(f"Expected a ROCm tensor, got {input_tensor.device}")
        if input_tensor.dtype != torch.float32:
            raise ValueError(f"Expected float32 input, got {input_tensor.dtype}")
        if tuple(input_tensor.shape) != expected:
            raise ValueError(
                f"Expected input shape {expected}, got {tuple(input_tensor.shape)}"
            )
        if not input_tensor.is_contiguous():
            raise ValueError("YOLO input must be contiguous")

        output = self.predictor.inference(input_tensor)
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise RuntimeError(f"Expected one YOLO output, got {len(output)}")
            output = output[0]
        if output.device.type != "cuda":
            raise RuntimeError(f"YOLO output left the GPU: {output.device}")
        if tuple(output.shape) != (1, 300, 6):
            raise RuntimeError(f"Unexpected YOLO output shape: {tuple(output.shape)}")
        return output

    def detect(self, blob):
        """Compatibility path for CPU-decoded frames; inference still runs on GPU."""
        import torch

        blob = np.ascontiguousarray(blob, dtype=np.float32)
        input_tensor = torch.from_numpy(blob).to(f"cuda:{self.device_id}")
        return self.infer_gpu(input_tensor).cpu().numpy()

    def detect_and_parse_gpu(self, input_tensor, scale, pad_w, pad_h, orig_shape):
        """Run Ultralytics MIGraphX inference and OpenCV GPU NMS without fallback."""
        output = self.infer_gpu(input_tensor)
        return self._parse_gpu(output, scale, pad_w, pad_h, orig_shape)

    def _parse_gpu(self, out_gpu, scale, pad_w, pad_h, orig_shape):
        """GPU post-processing: filter + un-letterbox + NMS, all on device.

        out_gpu: torch.Tensor (1, 300, 6) on CUDA — [x1, y1, x2, y2, score, class].
        Returns the same list-of-tuples format as _parse (CPU), so downstream
        overlay/VLM code is unchanged.
        """
        import torch

        dets = out_gpu[0]                      # (300, 6) on GPU
        scores = dets[:, 4]
        keep_mask = scores >= config.CONF_THRESHOLD
        dets = dets[keep_mask]
        if dets.shape[0] == 0:
            return []

        orig_h, orig_w = orig_shape[:2]

        # Un-letterbox coordinates on GPU, clamp to image bounds
        boxes = dets[:, :4].clone()
        boxes[:, 0] = ((boxes[:, 0] - pad_w) / scale).clamp(0, orig_w)
        boxes[:, 1] = ((boxes[:, 1] - pad_h) / scale).clamp(0, orig_h)
        boxes[:, 2] = ((boxes[:, 2] - pad_w) / scale).clamp(0, orig_w)
        boxes[:, 3] = ((boxes[:, 3] - pad_h) / scale).clamp(0, orig_h)

        # Drop degenerate boxes
        wh_ok = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        boxes = boxes[wh_ok]
        scores = dets[wh_ok, 4].contiguous()
        classes = dets[wh_ok, 5].to(torch.int32).contiguous()
        if boxes.shape[0] == 0:
            return []
        boxes = boxes.contiguous()

        n = boxes.shape[0]
        # Wrap the GPU tensors as GpuMats (zero copy) and run cv::cuda::nms
        gb = cv2.cuda_GpuMat.fromDevicePointer(boxes.data_ptr(), n, 4, cv2.CV_32FC1, 4 * 4)
        gs = cv2.cuda_GpuMat.fromDevicePointer(scores.data_ptr(), n, 1, cv2.CV_32FC1, 4)
        gc = cv2.cuda_GpuMat.fromDevicePointer(classes.data_ptr(), n, 1, cv2.CV_32SC1, 4)
        res = cv2.cuda.nms(gb, gs, gc, config.CONF_THRESHOLD, config.NMS_IOU_THRESHOLD)
        if res is None:
            return []
        idx = res.download().flatten() if isinstance(res, cv2.cuda.GpuMat) else np.asarray(res).flatten()
        if idx.size == 0:
            return []

        # Copy only survivors to host
        boxes_cpu = boxes[idx].cpu().numpy()
        scores_cpu = scores[idx].cpu().numpy()
        classes_cpu = classes[idx].cpu().numpy()
        results = []
        for k in range(len(idx)):
            x1, y1, x2, y2 = boxes_cpu[k]
            results.append((float(x1), float(y1), float(x2), float(y2),
                            float(scores_cpu[k]), int(classes_cpu[k])))
        return results

    def detect_and_parse(self, blob, scale, pad_w, pad_h, orig_shape):
        """Run detection and parse results into boxes in original image coords."""
        raw = self.detect(blob)  # (1, 300, 6)
        return self._parse(raw, scale, pad_w, pad_h, orig_shape)

    def _parse(self, raw, scale, pad_w, pad_h, orig_shape):
        """Parse raw (1, 300, 6) output into NMS-filtered boxes in orig coords."""
        detections = raw[0]  # (300, 6): x1, y1, x2, y2, score, class_id

        orig_h, orig_w = orig_shape[:2]
        results = []
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            if score < config.CONF_THRESHOLD:
                continue
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            results.append((x1, y1, x2, y2, float(score), int(class_id)))

        return self._nms(results)

    def _nms(self, detections):
        if not detections:
            return []
        boxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in detections]
        scores = [d[4] for d in detections]
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, config.CONF_THRESHOLD, config.NMS_IOU_THRESHOLD
        )
        if len(indices) == 0:
            return []
        return [detections[i] for i in indices.flatten()]
