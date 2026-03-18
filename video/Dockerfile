
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a

ENV VIRTUAL_ENV=/opt/venv
ARG DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update
RUN apt-get install --only-upgrade libc-bin libc6  && \
    apt-get install -y -q  --no-install-recommends python3-setuptools \
        python3-tornado curl libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY --from=lcc_base_video_image:latest ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=lcc_base_video_image:latest /home /home

# activate virtual environment
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory in the container
WORKDIR /home

COPY requirements.* /home/

ARG DEVICE="CPU"
ENV DEVICE="${DEVICE}"

RUN if [ "${DEVICE}" = "CPU" ]; then \
        pip3 install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" ; \
    fi;
# RUN pip3 install --no-cache-dir --require-hashes -r /home/requirements.txt && \
#     pip3 install --no-cache-dir --require-hashes -r /home/requirements.${DEVICE}.txt
RUN pip3 install --no-cache-dir -r /home/requirements.in && \
    pip3 install --no-cache-dir -r /home/requirements.${DEVICE}.txt

COPY *.py /home/
COPY *.yaml /home/
COPY *.conf /etc/nginx/
COPY cleanup.sh /home/
COPY manage.sh /home/
COPY frontend /home/frontend
# COPY include /home/include
CMD  ["/bin/bash","-c","python /home/resources/models/download_yolo.py && /home/manage.sh"]

####
ARG  USER
ARG  GROUP
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
## --no-log-init option can prevent the creation of large, sparse files in /var/log/lastlog and /var/log/faillog
RUN if [ ${GID} -gt 0 ]; then groupadd -f -g ${GID} ${GROUP}; fi; \
    if [ ${UID} -gt 0 ]; then useradd --no-log-init -d /home/${USER} -g ${GID} -K UID_MAX=${UID} -K UID_MIN=${UID} ${USER}; fi; \
    touch /var/run/nginx.pid && \
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf /home
VOLUME ["/var/www"]
USER ${UID}
####
