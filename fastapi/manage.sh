#!/bin/bash -e

/usr/local/sbin/nginx -g 'daemon on;'

# run fastapi
exec ${VIRTUAL_ENV}/bin/python /home/main.py
