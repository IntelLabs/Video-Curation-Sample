
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a as build

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv

ARG DEBUG="0"
ENV DEBUG="${DEBUG}"

RUN apt-get update
RUN apt-get install --only-upgrade libc-bin libc6 && \
    apt-get install -y -q  --no-install-recommends python3 python3-venv curl libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

COPY --from=lcc_base_video_image:latest ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=lcc_base_video_image:latest /home /home

# activate virtual environment
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG DEVICE="CPU"
ENV DEVICE="${DEVICE}"

RUN if [ "${DEVICE}" = "CPU" ]; then \
        pip3 install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" ; \
    fi;

# Set the working directory in the container
WORKDIR /home

COPY resources /home/resources
COPY requirements.* /home/
RUN python -m pip install pip --upgrade  --no-cache-dir  && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -r requirements.GPU.txt

COPY *.py /home/
COPY include /home/include
COPY templates /home/templates
COPY nginx.conf /etc/nginx/nginx.conf

# CMD /usr/local/sbin/nginx && uvicorn main:app --host 127.0.0.1 --port 8000
# EXPOSE 80 8000 8080
# CMD /usr/local/sbin/nginx -g 'daemon on;' && python3 /home/resources/models/download_yolo.py && python3 main.py
CMD  ["/bin/bash","-c","/usr/local/sbin/nginx -g 'daemon on;' && python /home/resources/models/download_yolo.py && python3 main.py"]
EXPOSE 80 8000 8080

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
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/logwatch && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf /home
VOLUME ["/var/www"]
USER ${UID}
####
