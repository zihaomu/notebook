"""YOLO26x object detection — MIGraphX GPU (zero-copy) or ONNX Runtime fallback."""

import os
import sys
import time

sys.path.insert(0, "/opt/opencv5/lib/python3.12/site-packages")
if "/opt/rocm/lib" not in sys.path:
    sys.path.insert(0, "/opt/rocm/lib")

import cv2
import numpy as np

import config


class YOLODetector:
    def __init__(self, model_path=None, compiled_path=None, device_id=0):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.compiled_path = compiled_path or config.YOLO_COMPILED_PATH
        self.device_id = device_id
        self.backend = None
        self._mgx_model = None
        self._mgx_input_name = None
        self._mgx_output_name = None
        self._mgx_input_shape = None
        self._mgx_output_shape = None
        self._mgx_output_tensor = None
        self._mgx_output_arg = None
        self._session = None
        self._load_model()

    def _load_model(self):
        backend_env = os.environ.get("YOLO_BACKEND", "auto")
        if backend_env in ("migraphx", "auto"):
            if self._try_migraphx():
                return
        if backend_env in ("ort", "auto"):
            self._try_onnxruntime()

    def _try_migraphx(self):
        try:
            import migraphx
            import torch
            if not torch.cuda.is_available():
                print("[detector] PyTorch CUDA not available, skipping MIGraphX")
                return False

            print("[detector] Trying MIGraphX GPU backend (zero-copy)...")

            if os.path.exists(self.compiled_path):
                t0 = time.time()
                self._mgx_model = migraphx.load(self.compiled_path)
                print(f"[detector] Loaded compiled model in {time.time()-t0:.2f}s")
            else:
                t0 = time.time()
                self._mgx_model = migraphx.parse_onnx(self.model_path)
                migraphx.quantize_fp16(self._mgx_model)
                self._mgx_model.compile(migraphx.get_target("gpu"), offload_copy=False)
                t1 = time.time()
                print(f"[detector] Compiled FP16 model in {t1-t0:.1f}s")
                migraphx.save(self._mgx_model, self.compiled_path)
                print(f"[detector] Saved to {self.compiled_path}")

            param_shapes = self._mgx_model.get_parameter_shapes()
            self._mgx_input_name = "images"
            self._mgx_output_name = next(
                n for n in param_shapes if n != self._mgx_input_name
            )
            self._mgx_input_shape = param_shapes[self._mgx_input_name]
            self._mgx_output_shape = param_shapes[self._mgx_output_name]

            self._mgx_output_tensor = torch.empty(
                self._mgx_output_shape.lens(),
                dtype=torch.float32, device="cuda",
            )
            self._mgx_output_arg = migraphx.argument_from_pointer(
                self._mgx_output_shape, self._mgx_output_tensor.data_ptr()
            )

            # Smoke test
            test_input = torch.randn(
                self._mgx_input_shape.lens(), dtype=torch.float32, device="cuda"
            )
            mgx_in = migraphx.argument_from_pointer(
                self._mgx_input_shape, test_input.data_ptr()
            )
            stream = torch.cuda.current_stream()
            self._mgx_model.run_async(
                {self._mgx_input_name: mgx_in,
                 self._mgx_output_name: self._mgx_output_arg},
                stream.cuda_stream, "ihipStream_t",
            )
            torch.cuda.synchronize()

            self.backend = "migraphx"
            print(f"[detector] MIGraphX GPU (FP16, zero-copy) ready")
            return True
        except Exception as e:
            print(f"[detector] MIGraphX failed: {e}")
            self._mgx_model = None
            return False

    def _try_onnxruntime(self):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        use = []
        if "ROCMExecutionProvider" in providers:
            use.append("ROCMExecutionProvider")
        use.append("CPUExecutionProvider")

        t0 = time.time()
        self._session = ort.InferenceSession(self.model_path, providers=use)
        active = self._session.get_providers()
        print(f"[detector] ONNX Runtime backend ready ({active[0]}) in {time.time()-t0:.2f}s")
        self.backend = "onnxruntime"

    def detect(self, blob):
        """Run YOLO inference.

        Args:
            blob: numpy array (1, 3, 640, 640) float32

        Returns:
            numpy array (1, 300, 6)
        """
        if self.backend == "migraphx":
            return self._detect_migraphx(blob)
        return self._detect_ort(blob)

    def _detect_migraphx(self, blob):
        import migraphx
        import torch

        blob = np.ascontiguousarray(blob, dtype=np.float32)
        input_tensor = torch.from_numpy(blob).cuda()

        mgx_in = migraphx.argument_from_pointer(
            self._mgx_input_shape, input_tensor.data_ptr()
        )
        stream = torch.cuda.current_stream()
        self._mgx_model.run_async(
            {self._mgx_input_name: mgx_in,
             self._mgx_output_name: self._mgx_output_arg},
            stream.cuda_stream, "ihipStream_t",
        )
        torch.cuda.synchronize()
        return self._mgx_output_tensor.cpu().numpy()

    def _detect_ort(self, blob):
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        return self._session.run(None, {"images": blob})[0]

    def _detect_migraphx_gpu_tensor(self, input_tensor):
        """Run MIGraphX on a GPU torch tensor and return the GPU output tensor.

        Args:
            input_tensor: torch.Tensor (1, 3, H, W) float32 on CUDA, contiguous.
        Returns:
            torch.Tensor (1, 300, 6) float32 on CUDA (the resident output buffer).
        """
        import migraphx
        import torch

        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        mgx_in = migraphx.argument_from_pointer(
            self._mgx_input_shape, input_tensor.data_ptr()
        )
        stream = torch.cuda.current_stream()
        self._mgx_model.run_async(
            {self._mgx_input_name: mgx_in,
             self._mgx_output_name: self._mgx_output_arg},
            stream.cuda_stream, "ihipStream_t",
        )
        torch.cuda.synchronize()
        return self._mgx_output_tensor

    def detect_and_parse_gpu(self, input_tensor, scale, pad_w, pad_h, orig_shape):
        """Zero-copy detection from a GPU input tensor, then GPU post-processing.

        Keeps the detection output on the GPU and runs confidence filtering,
        letterbox coordinate mapping, and NMS on the GPU (torch + cv::cuda::nms),
        copying only the surviving detections back to the host.

        Falls back to the numpy path if MIGraphX/GPU NMS isn't available.
        """
        if self.backend != "migraphx":
            raw = self.detect(input_tensor.cpu().numpy())
            return self._parse(raw, scale, pad_w, pad_h, orig_shape)

        out_gpu = self._detect_migraphx_gpu_tensor(input_tensor)
        try:
            return self._parse_gpu(out_gpu, scale, pad_w, pad_h, orig_shape)
        except Exception as e:
            if not getattr(self, "_gpu_nms_warned", False):
                print(f"[detector] GPU post-processing unavailable ({e}); using CPU path")
                self._gpu_nms_warned = True
            return self._parse(out_gpu.cpu().numpy(), scale, pad_w, pad_h, orig_shape)

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
