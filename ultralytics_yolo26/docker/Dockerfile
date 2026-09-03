ARG BASE_IMAGE=crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:opencv_end2end_2026_08_12
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID=unknown
ARG ULTRALYTICS_COMMIT=34e213ca3ece4c18962f5bb922ec74da0c474d24
ARG ULTRALYTICS_VERSION=8.4.75
ARG ONNXRUNTIME_MIGRAPHX_VERSION=1.24.2
ARG ULTRALYTICS_PATCH_SHA256=unknown
ARG HIP_VAAPI_BRIDGE_SHA256=unknown

LABEL org.opencontainers.image.title="Ultralytics YOLO26x AMD Radeon workshop" \
      org.opencontainers.image.description="Ultralytics ONNX with MIGraphX FP16 GPU I/O binding, OpenCV HIP video processing, and Jupyter" \
      io.amd.base-image.id="${BASE_IMAGE_ID}" \
      io.ultralytics.commit="${ULTRALYTICS_COMMIT}" \
      io.ultralytics.version="${ULTRALYTICS_VERSION}" \
      io.ultralytics.migraphx-iobinding-patch.sha256="${ULTRALYTICS_PATCH_SHA256}" \
      io.onnxruntime.migraphx.version="${ONNXRUNTIME_MIGRAPHX_VERSION}" \
      io.ultralytics.hip-vaapi-bridge.sha256="${HIP_VAAPI_BRIDGE_SHA256}" \
      io.ultralytics.workshop.workspace="/workspace"

COPY --from=ultralytics_source . /tmp/ultralytics
COPY --from=ort_wheel . /tmp/ort-wheel
COPY docker/patches/ultralytics-migraphx-iobinding.patch /tmp/ultralytics-migraphx-iobinding.patch
COPY docker/start-jupyter.sh /usr/local/bin/start-ultralytics-yolo26-jupyter
COPY docker/validate-image.py /usr/local/bin/validate-ultralytics-yolo26-image
COPY native/hip_vaapi_bridge.cpp /tmp/hip-vaapi-bridge/native/hip_vaapi_bridge.cpp
COPY scripts/build_native_bridge.sh /tmp/hip-vaapi-bridge/scripts/build_native_bridge.sh

RUN set -eux; \
    patch -d /tmp/ultralytics -p1 < /tmp/ultralytics-migraphx-iobinding.patch; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps \
        /tmp/ort-wheel/onnxruntime_migraphx-*.whl; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps \
        --no-build-isolation /tmp/ultralytics; \
    PACKAGE_ROOT=/tmp/hip-vaapi-bridge \
        OUTPUT_DIR=/opt/venv/lib/python3.10/site-packages \
        bash /tmp/hip-vaapi-bridge/scripts/build_native_bridge.sh; \
    /opt/venv/bin/python3 -c \
        "import cv2,hip_vaapi_bridge,numpy,onnxruntime,ultralytics; assert ultralytics.__version__ == '${ULTRALYTICS_VERSION}'; assert numpy.__version__ == '1.26.4'; assert cv2.__version__ == '5.1.0-dev'; assert onnxruntime.__version__ == '${ONNXRUNTIME_MIGRAPHX_VERSION}'"; \
    rm -rf /tmp/ultralytics /tmp/ort-wheel /tmp/hip-vaapi-bridge \
        /tmp/ultralytics-migraphx-iobinding.patch; \
    chmod 0755 /usr/local/bin/start-ultralytics-yolo26-jupyter \
        /usr/local/bin/validate-ultralytics-yolo26-image

ENV ULTRALYTICS_COMMIT=${ULTRALYTICS_COMMIT} \
    ULTRALYTICS_MIGRAPHX_PATCH_SHA256=${ULTRALYTICS_PATCH_SHA256} \
    ULTRALYTICS_HIP_VAAPI_BRIDGE_SHA256=${HIP_VAAPI_BRIDGE_SHA256} \
    ONNXRUNTIME_MIGRAPHX_VERSION=${ONNXRUNTIME_MIGRAPHX_VERSION} \
    ULTRALYTICS_YOLO26_ROOT=/workspace \
    ULTRALYTICS_YOLO26_MODEL_DIR=/workspace/models \
    ULTRALYTICS_YOLO26_OUTPUT_DIR=/workspace/output \
    ULTRALYTICS_MIGRAPHX_CACHE_ROOT=/workspace/models/ort-migraphx-cache \
    OPENCV_AMD_END2END_ROOT=/workspace \
    OPENCV_AMD_END2END_MODEL_DIR=/workspace/models \
    OPENCV_AMD_END2END_OUTPUT_DIR=/workspace/output \
    PYTHONPATH=/opt/opencv5/lib/python3.10/site-packages:/opt/rocm/lib:/workspace:/workspace/src:/workspace/scripts \
    LD_LIBRARY_PATH=/opt/opencv5/lib:/opt/rocm/lib:/usr/local/lib

WORKDIR /workspace
EXPOSE 8888

CMD ["/usr/local/bin/start-ultralytics-yolo26-jupyter"]
