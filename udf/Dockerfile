
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1

ENV VIRTUAL_ENV=/opt/venv
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y -q  --no-install-recommends python3-setuptools \
    python3-tornado python3-requests python3-ply python3-pip python3-psutil python3-venv \
    kafkacat curl libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY remote_function /home/remote_function
COPY requirements.txt /home/requirements.txt

RUN pip3 install pip --upgrade && pip3 install -r /home/requirements.txt

WORKDIR /home
COPY *.py /home/
COPY manage.sh /home/
CMD  ["/bin/bash","-c","/home/manage.sh"]

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
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/streams && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf /home/remote_function
# VOLUME ["/var/www"]
USER ${UID}
####
