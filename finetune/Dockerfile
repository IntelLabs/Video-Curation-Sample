FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV VIRTUAL_ENV=/opt/venv
ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary build tools and dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    gfortran \
    python3-dev \
    python3-numpy \
    python3-pip  \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.* /home/
RUN pip3 install pip --no-cache-dir --upgrade && \
    pip3 install --no-cache-dir --require-hashes -r /home/requirements.txt

# Define environment variables for OpenCV version and installation path
ENV OPENCV_VERSION="4.10.0"
ENV PYTHON_VERSION="3.$(${VIRTUAL_ENV}/bin/python -V | cut -f 1 | cut -d '.' -f 2)"
ENV DEPENDENCY_DIR=/tmp/build_opencv
ENV PYTHONPATH=${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages:${DEPENDENCY_DIR}/opencv/modules

# Working directory for source code
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
        -D CMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} \
        -D CUDA_ARCH_BIN=70,75,80,86,89,90 \
        -D CUDA_FAST_MATH=ON \
        -D ENABLE_FAST_MATH=ON \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=${DEPENDENCY_DIR}/opencv_contrib/modules \
        -D PYTHON_DEFAULT_EXECUTABLE=${VIRTUAL_ENV}/bin/python \
        -D PYTHON3_EXECUTABLE=${VIRTUAL_ENV}/bin/python \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=$(${VIRTUAL_ENV}/bin/python -c "import numpy; print(numpy.get_include())") \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_NVCUVENC=ON \
        -D WITH_NVCUVID=ON \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Clean up
RUN rm -rf ${DEPENDENCY_DIR}

# Set environment variables for installed OpenCV
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${VIRTUAL_ENV}/lib

WORKDIR /home

ENTRYPOINT python3 /home/finetune_test.py 2>&1 | tee /home/finetune.log