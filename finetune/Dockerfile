
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a AS build

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --only-upgrade --no-install-recommends libc-bin libc6 && \
    apt-get install -y -q  --no-install-recommends python3-setuptools \
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
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN python3 -m venv ${VIRTUAL_ENV}

ENV DEVICE="GPU"

# hadolint ignore=DL3008
RUN if [ "${DEVICE}" = "GPU" ]; then \
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb; \
        dpkg -i /tmp/cuda-keyring.deb; \
        rm /tmp/cuda-keyring.deb; \
        apt-get update; \
        apt-get install -y -q --no-install-recommends cuda-toolkit-12-4 libcudnn9-cuda-12 libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10; \
        rm -rf /var/lib/apt/lists/*; \
    fi;


ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# RUN pip3 install --no-cache-dir "pip==26.0.1" "torch==2.10.0" "torchvision==0.25.0" "numpy==1.26.4"
WORKDIR /home

COPY requirements.* /home/
# RUN pip3 install --no-cache-dir -r requirements.in
RUN pip3 install --no-cache-dir --require-hashes -r requirements.txt


# ARG PYTHON_VERSION=3.10
ENV PYTHON_VERSION=3.10
# OPENCV W/ CUDA SUPPORT
ENV OPENCV_VERSION="4.11.0"
# ENV PYTHON_VERSION="3.$(${VIRTUAL_ENV}/bin/python -V | cut -f 1 | cut -d '.' -f 2)"
ENV DEPENDENCY_DIR=/tmp/build_opencv
# ENV PYTHONPATH=${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages:${DEPENDENCY_DIR}/opencv/modules
ENV NUMPY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include"
ENV SYS_PATH="/usr/include/python${PYTHON_VERSION}"
WORKDIR ${DEPENDENCY_DIR}

# Clone OpenCV and OpenCV Contrib repositories
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git ${DEPENDENCY_DIR}/opencv&& \
    git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git ${DEPENDENCY_DIR}/opencv_contrib

# Create build directory and run CMake
WORKDIR ${DEPENDENCY_DIR}/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_JAVA=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_TESTS=OFF \
        -D CMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
        -D CUDA_ARCH_BIN=70,75,80,86,89,90 \
        -D CUDA_FAST_MATH=ON \
        -D ENABLE_FAST_MATH=ON \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH="${DEPENDENCY_DIR}/opencv_contrib/modules" \
        -D PYTHON_DEFAULT_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
        -D PYTHON3_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
        -D PYTHON3_NUMPY_INCLUDE_DIRS="${NUMPY_PATH}" \
        -D PYTHON3_PACKAGES_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages" \
        -D PYTHON3_LIBRARY="${VIRTUAL_ENV}/lib/libpython${PYTHON_VERSION}.so" \
        -D PYTHON3_INCLUDE_DIR="${SYS_PATH}" \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_NVCUVENC=ON \
        -D WITH_NVCUVID=ON \
        .. && \
    make "-j$(nproc)" && \
    make install && \
    ldconfig

# # Verify where it actually installed
# RUN find ${VIRTUAL_ENV} -name "cv2*.so"
# # Test the import immediately
# RUN ${VIRTUAL_ENV}/bin/python -c "import cv2; print(cv2.__version__)"

# Clean up
RUN rm -rf ${DEPENDENCY_DIR}

# Set the working directory in the container
# WORKDIR /home

# COPY requirements.* /home/
# # RUN pip3 install --no-cache-dir -r requirements.in
# RUN pip3 install --no-cache-dir --require-hashes -r requirements.txt


RUN \
    # Dynamically find the site-packages folder without hardcoding '3.10'
    PY_SITEPACKAGES=$(find "${VIRTUAL_ENV}/lib/" -maxdepth 2 -name "site-packages") && \
    echo "Found site-packages at: ${PY_SITEPACKAGES}" && \
    \
    # Link TensorRT libs
    ln -sf "${PY_SITEPACKAGES}/tensorrt_libs/lib*" /usr/lib/ && \
    \
    # Link NVIDIA libs
    find "${PY_SITEPACKAGES}/nvidia/" -name "lib*.so*" -exec ln -sf {} /usr/lib/ \; && \
    \
    ldconfig;

ARG DEBUG="0"
ENV DEBUG="${DEBUG}"
