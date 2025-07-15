
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1

ARG DEVICE="CPU"
ENV DEVICE="${DEVICE}"

ARG IN_SOURCE="videos"
ENV IN_SOURCE="${IN_SOURCE}"

ARG RESIZE_FLAG="False"
ENV RESIZE_FLAG="${RESIZE_FLAG}"

ENV VIRTUAL_ENV=/opt/venv
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y -q  --no-install-recommends python3-setuptools \
    python3-tornado python3-requests python3-ply python3-pip python3-psutil python3-venv \
    kafkacat curl libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY remote_function /home/remote_function
COPY resources /home/resources

RUN pip3 install pip --upgrade && \
    if [ "${DEVICE}" = "CPU" ]; then \
        pip3 install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision ultralytics "numpy>=1.23.5,<2.0.0"; \
    else \
        pip3 install torch torchvision ultralytics "numpy>=1.23.5,<2.0.0"; \
    fi;
RUN pip3 install wheel 'kafka-python>=1.4.7' inotify vdms && \
    pip3 install -r /home/remote_function/requirements.${DEVICE}.txt

RUN omz_downloader --list /home/resources/models/models.lst -o /home/resources/models --precisions FP16

# WORKDIR /var/www/mp4
# RUN rm /var/www/mp4/* || true
# RUN rm /var/www/stream/* || true
COPY archive/*.mp4 /var/
# RUN rm /var/www/mp4/.gitignore || true

WORKDIR /home
COPY *.py /home/
COPY cleanup.sh /home/
COPY manage.sh /home/
COPY *.conf /etc/nginx/
COPY segment_archive.py /home/
# RUN cp -rp /var/www/archive/*.mp4 /var/www/mp4/ || true  && \
#     rm -rf /var/www/archive && \
#     if [ "${IN_SOURCE}" = "videos" ]; then python3 /home/segment_archive.py; fi;
CMD  ["/bin/bash","-c","python3 /home/resources/models/download_yolo.py && /home/manage.sh"]
# CMD  ["/bin/bash","-c","/home/manage.sh"]

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
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/streams /home/resources/models/ultralytics/yolo11/FP16 && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var /etc/nginx/nginx.conf /home/resources /home/remote_function
VOLUME ["/var/www"]
USER ${UID}
####
