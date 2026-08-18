ARG BASE_IMAGE=zihao/opencv-llamacpp-q8:rocm7.2.1
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID=unknown
ARG OPENCV_COMMIT=unknown
ARG OPENCV_CONTRIB_COMMIT=unknown
ARG JUPYTERLAB_VERSION=4.6.0

LABEL org.opencontainers.image.title="OpenCV AMD end-to-end notebook runtime" \
      org.opencontainers.image.description="OpenCV 5 HIP, rocDecode, MIGraphX, and Jupyter with /workspace as the user mount" \
      io.amd.base-image.id="${BASE_IMAGE_ID}" \
      io.opencv.source.commit="${OPENCV_COMMIT}" \
      io.opencv.contrib.commit="${OPENCV_CONTRIB_COMMIT}" \
      io.jupyterlab.version="${JUPYTERLAB_VERSION}" \
      io.opencv.amd.workspace="/workspace" \
      io.opencv.amd.tooling="/opencv_workspace"

# These named contexts are supplied by scripts/build_notebook_image.sh. The
# installed runtime already lives in /opt; the source trees are retained for
# inspection and rebuilding without a bind mount to the legacy host workspace.
COPY --from=opencv_source --exclude=.git --exclude=.cache . /opencv_workspace/third_party/opencv
COPY --from=opencv_contrib_source --exclude=.git . /opencv_workspace/third_party/opencv_contrib

COPY docker/start-jupyter.sh /usr/local/bin/start-opencv-amd-jupyter
COPY docker/validate-image.py /usr/local/bin/validate-opencv-amd-image

RUN set -eux; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir "jupyterlab==${JUPYTERLAB_VERSION}"; \
    mkdir -p /workspace /opencv_workspace/opencv_end2end; \
    ln -s ../third_party /opencv_workspace/opencv_end2end/third_party; \
    chmod 0755 /usr/local/bin/start-opencv-amd-jupyter /usr/local/bin/validate-opencv-amd-image; \
    printf '%s\n' \
        "base_image_id=${BASE_IMAGE_ID}" \
        "opencv_commit=${OPENCV_COMMIT}" \
        "opencv_contrib_commit=${OPENCV_CONTRIB_COMMIT}" \
        > /opencv_workspace/THIRD_PARTY_VERSIONS

ENV OPENCV_AMD_END2END_ROOT=/workspace \
    OPENCV_AMD_END2END_MODEL_DIR=/workspace/models \
    OPENCV_AMD_END2END_OUTPUT_DIR=/workspace/output \
    OPENCV_INSTALL=/opt/opencv5 \
    OPENCV_SOURCE_DIR=/opencv_workspace/third_party/opencv \
    OPENCV_CONTRIB_SOURCE_DIR=/opencv_workspace/third_party/opencv_contrib \
    PYTHONPATH=/opt/opencv5/lib/python3.10/site-packages:/opt/rocm/lib:/workspace:/workspace/src:/workspace/scripts \
    LD_LIBRARY_PATH=/opt/opencv5/lib:/opt/rocm/lib:/usr/local/lib

WORKDIR /workspace
EXPOSE 8888

CMD ["/usr/local/bin/start-opencv-amd-jupyter"]
