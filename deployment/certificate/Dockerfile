
FROM alpine:3.16@sha256:452e7292acee0ee16c332324d7de05fa2c99f9994ecc9f0779c602916a672ae4
RUN apk add --no-cache --upgrade openssl bash && rm -rf /var/cache/apk/*


####
ARG  USER
ARG  GROUP
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
RUN if [ ${GID} -gt 0 ]; then addgroup -g ${GID} ${GROUP}; fi; \
    if [ ${UID} -gt 0 ]; then adduser -h /home/${USER} -G ${GROUP} -u ${UID} ${USER}; fi; \
     echo
USER ${UID}
####
