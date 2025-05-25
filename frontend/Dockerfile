
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1

ARG DEBUG="0"
ENV DEBUG="${DEBUG}"

RUN apt-get update && apt-get install -y -q --no-install-recommends python3-tornado python3-requests python3-ply python3-pip python3-psutil && rm -rf /var/lib/apt/lists/*
RUN pip3 install 'protobuf==4.24.2' vdms

COPY *.py /home/
COPY    *.conf /etc/nginx/
COPY    html /var/www/html
CMD ["/bin/bash","-c","/home/frontend.py&/usr/local/sbin/nginx"]
EXPOSE  8080

####
ARG  USER
ARG  GROUP
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
RUN if [ ${GID} -gt 0 ]; then groupadd -f -g ${GID} ${GROUP}; fi; \
    if [ ${UID} -gt 0 ]; then useradd --no-log-init -d /home/${USER} -g ${GID} -K UID_MAX=${UID} -K UID_MIN=${UID} ${USER}; fi; \
    touch /var/run/nginx.pid && \
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/logwatch && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf
VOLUME ["/var/www"]
USER ${UID}
####
