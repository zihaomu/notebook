ARG BASE_IMAGE=crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:opencv_end2end_2026_08_12
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID=unknown
ARG ULTRALYTICS_COMMIT=34e213ca3ece4c18962f5bb922ec74da0c474d24
ARG ULTRALYTICS_VERSION=8.4.75
ARG ONNXRUNTIME_MIGRAPHX_VERSION=1.24.2
ARG ULTRALYTICS_PATCH_SHA256=unknown
ARG HIP_VAAPI_BRIDGE_SHA256=unknown
ARG WORKSHOP_BUNDLE_SHA256=unknown
ARG YOLO_CHECKPOINT_SHA256=unknown
ARG YOLO_ONNX_SHA256=unknown
ARG MIGRAPHX_CACHE_SHA256=unknown
ARG QWEN_GGUF_SHA256=unknown
ARG QWEN_MMPROJ_SHA256=unknown
ARG MODEL_SET_SHA256=unknown
ARG WORKSHOP_GIT_COMMIT=unknown

LABEL org.opencontainers.image.title="Ultralytics YOLO26x AMD Radeon baked workshop" \
      org.opencontainers.image.description="Self-contained Ultralytics workshop with baked YOLO26x, gfx1100 MIGraphX cache, HIP/VAAPI binaries, and Jupyter" \
      org.opencontainers.image.source="https://gh-test.anruicloud.com/zihaomu/notebook" \
      org.opencontainers.image.revision="${WORKSHOP_GIT_COMMIT}" \
      io.amd.base-image.id="${BASE_IMAGE_ID}" \
      io.ultralytics.commit="${ULTRALYTICS_COMMIT}" \
      io.ultralytics.version="${ULTRALYTICS_VERSION}" \
      io.ultralytics.migraphx-iobinding-patch.sha256="${ULTRALYTICS_PATCH_SHA256}" \
      io.onnxruntime.migraphx.version="${ONNXRUNTIME_MIGRAPHX_VERSION}" \
      io.ultralytics.hip-vaapi-bridge.sha256="${HIP_VAAPI_BRIDGE_SHA256}" \
      io.ultralytics.workshop.bundle="baked" \
      io.ultralytics.workshop.bundle.sha256="${WORKSHOP_BUNDLE_SHA256}" \
      io.ultralytics.yolo26x.checkpoint.sha256="${YOLO_CHECKPOINT_SHA256}" \
      io.ultralytics.yolo26x.onnx.sha256="${YOLO_ONNX_SHA256}" \
      io.ultralytics.migraphx-cache.sha256="${MIGRAPHX_CACHE_SHA256}" \
      io.ultralytics.qwen-gguf.sha256="${QWEN_GGUF_SHA256}" \
      io.ultralytics.qwen-mmproj.sha256="${QWEN_MMPROJ_SHA256}" \
      io.ultralytics.model-set.sha256="${MODEL_SET_SHA256}" \
      io.ultralytics.qwen-models="baked" \
      io.ultralytics.models.path="/opt/ultralytics-yolo26/models" \
      io.ultralytics.workshop.workspace="/workspace"

COPY --from=ultralytics_source . /tmp/ultralytics
COPY --from=ort_wheel . /tmp/ort-wheel
COPY docker/patches/ultralytics-migraphx-iobinding.patch /tmp/ultralytics-migraphx-iobinding.patch
COPY docker/start-jupyter.sh /usr/local/bin/start-ultralytics-yolo26-jupyter
COPY docker/validate-image.py /usr/local/bin/validate-ultralytics-yolo26-image
COPY docker/init-model-volume.sh /usr/local/bin/init-ultralytics-yolo26-model-volume
COPY native/hip_vaapi_bridge.cpp /tmp/hip-vaapi-bridge/native/hip_vaapi_bridge.cpp
COPY native/vaapi_hip_encode_probe.cpp /tmp/hip-vaapi-bridge/native/vaapi_hip_encode_probe.cpp
COPY scripts/build_native_bridge.sh /tmp/hip-vaapi-bridge/scripts/build_native_bridge.sh
COPY scripts/build_vaapi_hip_probe.sh /tmp/hip-vaapi-bridge/scripts/build_vaapi_hip_probe.sh

RUN set -eux; \
    patch -d /tmp/ultralytics -p1 < /tmp/ultralytics-migraphx-iobinding.patch; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps \
        /tmp/ort-wheel/onnxruntime_migraphx-*.whl; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps \
        --no-build-isolation /tmp/ultralytics; \
    PACKAGE_ROOT=/tmp/hip-vaapi-bridge \
        OUTPUT_DIR=/opt/venv/lib/python3.10/site-packages \
        bash /tmp/hip-vaapi-bridge/scripts/build_native_bridge.sh; \
    PACKAGE_ROOT=/tmp/hip-vaapi-bridge \
        OUTPUT=/usr/local/bin/vaapi-hip-encode-probe \
        bash /tmp/hip-vaapi-bridge/scripts/build_vaapi_hip_probe.sh; \
    /opt/venv/bin/python3 -c \
        "import cv2,hip_vaapi_bridge,numpy,onnxruntime,ultralytics; assert ultralytics.__version__ == '${ULTRALYTICS_VERSION}'; assert numpy.__version__ == '1.26.4'; assert cv2.__version__ == '5.1.0-dev'; assert onnxruntime.__version__ == '${ONNXRUNTIME_MIGRAPHX_VERSION}'"; \
    rm -rf /tmp/ultralytics /tmp/ort-wheel /tmp/hip-vaapi-bridge \
        /tmp/ultralytics-migraphx-iobinding.patch; \
    chmod 0755 /usr/local/bin/start-ultralytics-yolo26-jupyter \
        /usr/local/bin/validate-ultralytics-yolo26-image

COPY .dockerignore .gitignore README.md README_CN.md \
    ultralytics_yolo26x_step_by_step.ipynb \
    ultralytics_yolo26x_end_to_end.ipynb \
    /opt/ultralytics-yolo26/seed/
COPY assets /opt/ultralytics-yolo26/seed/assets
COPY data /opt/ultralytics-yolo26/seed/data
COPY doc /opt/ultralytics-yolo26/seed/doc
COPY docker /opt/ultralytics-yolo26/seed/docker
COPY native /opt/ultralytics-yolo26/seed/native
COPY output /opt/ultralytics-yolo26/seed/output
COPY scripts /opt/ultralytics-yolo26/seed/scripts
COPY src /opt/ultralytics-yolo26/seed/src
COPY tests /opt/ultralytics-yolo26/seed/tests
RUN set -eux; \
    computed_bundle_sha256="$(/opt/venv/bin/python3 \
        /opt/ultralytics-yolo26/seed/scripts/workshop_bundle_identity.py \
        /opt/ultralytics-yolo26/seed)"; \
    test "$computed_bundle_sha256" = "${WORKSHOP_BUNDLE_SHA256}"; \
    mkdir -p /workspace; \
    cp -al /opt/ultralytics-yolo26/seed/. /workspace/; \
    /opt/venv/bin/python3 -m compileall -q \
        /workspace/src /workspace/scripts /workspace/tests; \
    test -x /usr/local/bin/vaapi-hip-encode-probe; \
    test -x /usr/local/bin/init-ultralytics-yolo26-model-volume; \
    printf '%s\n' "${WORKSHOP_BUNDLE_SHA256}" \
        > /usr/local/share/ultralytics-yolo26-bundle.sha256; \
    chmod -R a+rX /opt/ultralytics-yolo26/seed; \
    chmod -R a+rwX \
        /workspace/assets /workspace/data /workspace/doc /workspace/docker \
        /workspace/native /workspace/output /workspace/scripts /workspace/src \
        /workspace/tests /workspace/*.md /workspace/*.ipynb; \
    chmod 0777 /workspace

COPY --link models /opt/ultralytics-yolo26/models

RUN set -eux; \
    ln -s /opt/ultralytics-yolo26/models /workspace/models; \
    cd /opt/ultralytics-yolo26/models; \
    test "$(sha256sum yolo26x.pt | cut -d' ' -f1)" = "${YOLO_CHECKPOINT_SHA256}"; \
    test "$(sha256sum yolo26x.onnx | cut -d' ' -f1)" = "${YOLO_ONNX_SHA256}"; \
    test "$(sha256sum Qwen3-VL-8B-Instruct-Q8_0.gguf | cut -d' ' -f1)" = "${QWEN_GGUF_SHA256}"; \
    test "$(sha256sum mmproj-F16.gguf | cut -d' ' -f1)" = "${QWEN_MMPROJ_SHA256}"; \
    test "$(sha256sum ort-migraphx-cache/735f1583e99dfeb733da/20e00-58de11c69ae52cf2-9880cf1608079e0d-36a8840bfe2de0d1.mxr | cut -d' ' -f1)" = "${MIGRAPHX_CACHE_SHA256}"; \
    sha256sum \
        yolo26x.pt yolo26x.onnx Qwen3-VL-8B-Instruct-Q8_0.gguf mmproj-F16.gguf \
        ort-migraphx-cache/735f1583e99dfeb733da/identity.json \
        ort-migraphx-cache/735f1583e99dfeb733da/20e00-58de11c69ae52cf2-9880cf1608079e0d-36a8840bfe2de0d1.mxr \
        > SHA256SUMS; \
    test "$(sha256sum SHA256SUMS | cut -d' ' -f1)" = "${MODEL_SET_SHA256}"; \
    printf '%s\n' "${MODEL_SET_SHA256}" > MODEL_SET_ID

ENV ULTRALYTICS_COMMIT=${ULTRALYTICS_COMMIT} \
    ULTRALYTICS_MIGRAPHX_PATCH_SHA256=${ULTRALYTICS_PATCH_SHA256} \
    ULTRALYTICS_HIP_VAAPI_BRIDGE_SHA256=${HIP_VAAPI_BRIDGE_SHA256} \
    ULTRALYTICS_WORKSHOP_BUNDLE=baked \
    ULTRALYTICS_WORKSHOP_BUNDLE_SHA256=${WORKSHOP_BUNDLE_SHA256} \
    ULTRALYTICS_YOLO26_CHECKPOINT_SHA256=${YOLO_CHECKPOINT_SHA256} \
    ULTRALYTICS_YOLO26_ONNX_SHA256=${YOLO_ONNX_SHA256} \
    ULTRALYTICS_MIGRAPHX_CACHE_SHA256=${MIGRAPHX_CACHE_SHA256} \
    ULTRALYTICS_QWEN_GGUF_SHA256=${QWEN_GGUF_SHA256} \
    ULTRALYTICS_QWEN_MMPROJ_SHA256=${QWEN_MMPROJ_SHA256} \
    ULTRALYTICS_MODEL_SET_SHA256=${MODEL_SET_SHA256} \
    ULTRALYTICS_BAKED_MODEL_DIR=/opt/ultralytics-yolo26/models \
    ONNXRUNTIME_MIGRAPHX_VERSION=${ONNXRUNTIME_MIGRAPHX_VERSION} \
    ULTRALYTICS_YOLO26_ROOT=/workspace \
    ULTRALYTICS_YOLO26_MODEL_DIR=/opt/ultralytics-yolo26/models \
    ULTRALYTICS_YOLO26_OUTPUT_DIR=/workspace/output \
    ULTRALYTICS_MIGRAPHX_CACHE_ROOT=/opt/ultralytics-yolo26/models/ort-migraphx-cache \
    OPENCV_AMD_END2END_ROOT=/workspace \
    OPENCV_AMD_END2END_MODEL_DIR=/opt/ultralytics-yolo26/models \
    OPENCV_AMD_END2END_OUTPUT_DIR=/workspace/output \
    PYTHONPATH=/opt/opencv5/lib/python3.10/site-packages:/opt/rocm/lib:/workspace:/workspace/src:/workspace/scripts \
    LD_LIBRARY_PATH=/opt/opencv5/lib:/opt/rocm/lib:/usr/local/lib

WORKDIR /workspace
EXPOSE 8888

CMD ["/usr/local/bin/start-ultralytics-yolo26-jupyter"]
