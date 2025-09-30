#!/bin/bash -e

# Watch directory
python3 /home/watch_and_send2vdms.py ${WATCH_DIR} &

# run tornado
exec /home/manage.py
