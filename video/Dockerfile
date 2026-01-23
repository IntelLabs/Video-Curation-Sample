
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a as build

ENV VIRTUAL_ENV=/opt/venv
ARG DEBIAN_FRONTEND=noninteractive

# fixes CVE-2023-4911 vulnerability on Ubuntu 22.04
RUN apt-get update && \
    apt-get install --only-upgrade libc-bin libc6  && \
    apt-get install -y -q  --no-install-recommends python3-setuptools \
    python3-tornado python3-ply python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY resources /home/resources
COPY requirements.* /home/

ARG DEVICE="CPU"
ENV DEVICE="${DEVICE}"

RUN pip3 install pip --no-cache-dir --upgrade && \
    if [ "${DEVICE}" = "CPU" ]; then \
        pip3 install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" ; \
    fi;
RUN pip3 install --no-cache-dir --require-hashes -r /home/requirements.txt && \
    pip3 install --no-cache-dir --require-hashes -r /home/requirements.${DEVICE}.txt

RUN omz_downloader --list /home/resources/models/models.lst -o /home/resources/models --precisions FP16

# FINAL IMAGE
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a
ENV VIRTUAL_ENV=/opt/venv
ARG DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install --only-upgrade libc-bin libc6  && \
    apt-get install -y -q  --no-install-recommends python3-tornado \
        curl libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=build /home /home


WORKDIR /home
COPY *.py /home/
COPY cleanup.sh /home/
COPY manage.sh /home/
COPY *.conf /etc/nginx/
CMD  ["/bin/bash","-c","python3 /home/resources/models/download_yolo.py && /home/manage.sh"]

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
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf /home/resources
VOLUME ["/var/www"]
USER ${UID}
####
