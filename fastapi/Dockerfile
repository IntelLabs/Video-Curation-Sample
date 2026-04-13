
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
        cuda-toolkit-12-4 libcudnn9-cuda-12 libnvinfer10 \
        libnvonnxparsers10 libnvinfer-plugin10 && \
    rm -rf /var/lib/apt/lists/* && apt-get clean


RUN python3 -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir \
    "pip==26.0.1" \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "numpy==1.26.0"

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

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /tmp/cv2_deps && \
    PY_CV2_SO=$(find "${VIRTUAL_ENV}" -name "cv2*.so") && \
    ldd "${PY_CV2_SO}" | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' /tmp/cv2_deps/

# Clean up
RUN rm -rf ${DEPENDENCY_DIR}

FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install ONLY runtime shared libraries (No compilers/toolkits)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl python3 libgl1 libglib2.0-0 libgtk2.0-0 libtbb12 libjpeg8 libpng16-16 \
#     libavcodec58 libavformat58 libswscale5 libv4l-0 && \
#     # libcudnn9-cuda-12 && \
#     # Install necessary CUDA packages
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
#         dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
#     apt-get update && \
#     apt-get install -y -q --no-install-recommends \
#         libcudnn9-cuda-12 && \
#     rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3008
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl python3 libglib2.0-0 libtbb12 && \
#     rm -rf /var/lib/apt/lists/*

# Install Runtime Libraries, FFmpeg/VA-API headers, and cuDNN
# Includes libva and ffmpeg libraries required for HEVC decoding
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
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
    apt-get install -y -q --no-install-recommends libcudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire pre-compiled virtual environment from the build stage
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# Copy NPP (NVIDIA Performance Primitives) libs - Required for OpenCV CUDA
# COPY --from=build /usr/local/cuda/lib64/libnpp*.so* /usr/local/cuda/lib64/
# Copy additional required CUDA math/parallel libs
RUN mkdir -p /usr/local/cuda/lib64
COPY --from=build /tmp/cv2_deps/* /usr/local/cuda/lib64/


ARG DEVICE="CPU"
ENV DEVICE="${DEVICE}"

ARG DEBUG="0"
ENV DEBUG="${DEBUG}"

# Set the working directory in the container
WORKDIR /home
COPY requirements.* /home/

RUN if [ "${DEVICE}" = "CPU" ]; then \
        pip3 install --no-cache-dir --require-hashes -r /home/requirements.CPU.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple; \
    else \
        pip3 install --no-cache-dir --require-hashes -r requirements.GPU.txt; \
    fi;

COPY *.py /home/
COPY *.sh /home/
COPY include /home/include
COPY templates /home/templates
COPY nginx.conf /etc/nginx/nginx.conf

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the Entrypoint
ENTRYPOINT ["entrypoint.sh"]

# Your simplified CMD
CMD ["/bin/bash", "-c", "/home/manage.sh"]
# CMD  ["/bin/bash","-c","python /home/resources/models/download_yolo.py && /home/manage.sh"]

EXPOSE 80 8000 8080

####
ARG  USER
ARG  GROUP
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
## --no-log-init option can prevent the creation of large, sparse files in /var/log/lastlog and /var/log/faillog
# RUN if [ ${GID} -gt 0 ]; then groupadd -f -g ${GID} ${GROUP}; fi; \
#     if [ ${UID} -gt 0 ]; then useradd --no-log-init -d /home/${USER} -g ${GID} -u ${UID} ${USER}; fi; \
#     touch /var/run/nginx.pid && \
#     mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/logwatch && \
#     chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf /home
RUN if [ ${UID} -gt 0 ]; then \
        # Manually create group and user entries to bypass system limits
        echo "${GROUP}:x:${GID}:" >> /etc/group && \
        echo "${USER}:x:${UID}:${GID}::/home/${USER}:/bin/bash" >> /etc/passwd; \
        # mkdir -p /home/${USER} && \
        # chown -R ${UID}:${GID} /home/${USER}; \
    fi && \
    touch /var/run/nginx.pid && \
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/logwatch && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf /home
VOLUME ["/var/www"]
USER ${UID}
####
