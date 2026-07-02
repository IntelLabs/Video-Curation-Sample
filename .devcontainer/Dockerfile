
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a AS build

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --only-upgrade --no-install-recommends libc-bin libc6 && \
    apt-get install -y -q --no-install-recommends python3-setuptools \
        python3-dev python3-pip python3-venv \
        curl libgl1-mesa-glx \
        # Update and install necessary build tools and dependencies for OpenCV
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        libgtk2.0-dev \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libxvidcore-dev \
        libxine2-dev \
        libv4l-dev \
        libdc1394-dev \
        libatlas-base-dev \
        gfortran && \
    # Install necessary CUDA packages
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
        dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y -q --no-install-recommends \
        cuda-toolkit-12-4 \
        libcudnn9-dev-cuda-12 \
        libnpp-dev-12-4 \
        && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN python3 -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir \
    "pip==26.1.2" \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "numpy==1.26.4"

# OPENCV W/ CUDA SUPPORT
ENV PYTHON_VERSION=3.10
ENV OPENCV_VERSION="4.11.0"
ENV DEPENDENCY_DIR=/tmp/build_opencv
ENV NUMPY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include"
ENV SYS_PATH="/usr/include/python${PYTHON_VERSION}"

# Clone OpenCV and OpenCV Contrib repositories
WORKDIR ${DEPENDENCY_DIR}
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git ${DEPENDENCY_DIR}/opencv && \
    git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git ${DEPENDENCY_DIR}/opencv_contrib

# Create build directory and run CMake
WORKDIR ${DEPENDENCY_DIR}/opencv/build
RUN ln -s /usr/include/x86_64-linux-gnu/cudnn*.h /usr/local/cuda/include/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn*.so* /usr/local/cuda/lib64/
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -D OPENCV_EXTRA_MODULES_PATH="${DEPENDENCY_DIR}/opencv_contrib/modules" \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=70,75,80,86,89,90 \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
    -D BUILD_opencv_cudaimgproc=ON \
    -D BUILD_opencv_cudaarithm=ON \
    -D BUILD_opencv_cudafilters=ON \
    -D BUILD_opencv_cudacodec=ON \
    -D WITH_NVCUVID=ON \
    -D WITH_NVCUVENC=ON \
    -D WITH_VAAPI=ON \
    -D WITH_FFMPEG=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE="${VIRTUAL_ENV}/bin/python3" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="${VIRTUAL_ENV}/lib/python3.10/site-packages/numpy/core/include" \
    -D OPENCV_SKIP_PYTHON_LOADER=ON \
    -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF .. \
    && make -j$(nproc) && make install && ldconfig

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /tmp/cv2_deps && \
    PY_CV2_SO=$(find "${VIRTUAL_ENV}" -name "cv2*.so" | head -n 1) && \
    ldd "${PY_CV2_SO}" | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' /tmp/cv2_deps/ && \
    cp -v /usr/local/cuda/lib64/libnpp*.so* /tmp/cv2_deps/ && \
    cp -v /usr/local/cuda/lib64/libcudart.so* /tmp/cv2_deps/ && \
    cp -P /usr/local/cuda/lib64/libnvrtc.so* /tmp/cv2_deps/ && \
    cp -v /usr/lib/x86_64-linux-gnu/libnvidia-encode.so* /tmp/cv2_deps/ || true


# Clean up
RUN rm -rf ${DEPENDENCY_DIR}


FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:${PATH}"

# Install Runtime Libraries, FFmpeg/VA-API headers, and cuDNN
# Includes libva and ffmpeg libraries required for HEVC decoding
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    libgl1 \
    libglib2.0-0 \
    libtbb12 \
    # Critical for Hardware Decoding (HEVC)
    libva2 \
    libva-drm2 \
    libva-x11-2 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 && \
    # Install necessary CUDA packages
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
        dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y -q --no-install-recommends libcudnn9-cuda-12 \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev  nvidia-cuda-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire pre-compiled virtual environment from the build stage
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# Copy NPP (NVIDIA Performance Primitives) libs - Required for OpenCV CUDA
# COPY --from=build /usr/local/cuda/lib64/libnpp*.so* /usr/local/cuda/lib64/
# Copy additional required CUDA math/parallel libs
RUN mkdir -p /usr/local/cuda/lib64
COPY --from=build /tmp/cv2_deps/* /usr/local/cuda/lib64/
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
RUN ldconfig

ARG DEVICE="CPU"
ENV DEVICE="${DEVICE}"

# Set the working directory in the container
WORKDIR /home
COPY requirements.txt /home/
ENV NVIDIA_DRIVER_CAPABILITIES=all

# RUN pip3 install --no-cache-dir "pytest>=9.0.3" && \
#     pip3 install --no-cache-dir --require-hashes -r /home/requirements.CPU.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple && \
#     pip3 uninstall -y ultralytics opencv-python opencv-contrib-python opencv-python-headless && \
#     pip3 install --no-cache-dir --require-hashes -r /home/requirements.GPU.txt && \
#     pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless


RUN pip3 install --no-cache-dir --require-hashes -r /home/requirements.txt && \
    pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless