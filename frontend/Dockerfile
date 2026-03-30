
FROM openvisualcloud/xeon-ubuntu2204-media-nginx:23.1@sha256:d19eb597dc210134063803630ae2ea1ec84dfd4189138f59551e2f5ed047284a

ARG DEBUG="0"
ENV DEBUG="${DEBUG}"
ARG DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# fixes CVE-2023-4911 vulnerability on Ubuntu 22.04
RUN apt-get update
RUN apt-get install --only-upgrade libc-bin libc6  && \
    apt-get install -y -q --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.* /home/
RUN pip3 install --no-cache-dir --require-hashes -r /home/requirements.txt
# RUN pip3 install --no-cache-dir -r /home/requirements.in

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
## --no-log-init option can prevent the creation of large, sparse files in /var/log/lastlog and /var/log/faillog
RUN if [ ${GID} -gt 0 ]; then groupadd -f -g ${GID} ${GROUP}; fi; \
    if [ ${UID} -gt 0 ]; then useradd --no-log-init -d /home/${USER} -g ${GID} -K UID_MAX=${UID} -K UID_MIN=${UID} ${USER}; fi; \
    touch /var/run/nginx.pid && \
    mkdir -p /var/log/nginx /var/lib/nginx /var/www/cache /var/www/gen /var/www/mp4 /var/www/logwatch && \
    chown -R ${UID}.${GID} /var/run/nginx.pid /var/log/nginx /var/lib/nginx /var/www /etc/nginx/nginx.conf
VOLUME ["/var/www"]
USER ${UID}
####
