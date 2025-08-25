#!/bin/bash -e

# Watch directory
# python3 /home/watch_and_send2vdms.py /var/www/archive &  # Default
# python3 /home/watch_and_send2vdms_cb.py /var/www/archive &  # UDF for metadata
python3 /home/watch_and_send2vdms_cb.py ${WATCH_DIR} &

# run tornado
exec /home/manage.py
