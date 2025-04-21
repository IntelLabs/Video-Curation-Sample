
FROM openvisualcloud/xeon-ubuntu2004-media-nginx:23.1

RUN apt-get update && \
    apt-get install -y -q  --no-install-recommends python3-setuptools \
    python3-tornado python3-requests python3-ply python3-pip python3-psutil \
    kafkacat curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install 'kafka-python>=1.4.7' inotify

COPY *.py /home/
COPY cleanup.sh /home/
COPY manage.sh /home/
COPY *.conf /etc/nginx/
COPY archive /var/www/mp4
RUN rm /var/www/mp4/.gitignore || true
CMD  ["/bin/bash","-c","/home/manage.sh"]

####
ARG  USER
ARG  GROUP
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
RUN if [ ${GID} -gt 0 ]; then groupadd -f -g ${GID} ${GROUP}; fi; \
    if [ ${UID} -gt 0 ]; then useradd -d /home/${USER} -g ${GID} -K UID_MAX=${UID} -K UID_MIN=${UID} ${USER}; fi; \
     touch /var/run/nginx.pid && \
     mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/streams && \
     chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf
VOLUME ["/var/www"]
USER ${UID}
####
