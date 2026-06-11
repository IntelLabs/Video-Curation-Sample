#!/bin/bash -e

# Watch directory
echo "WATCH_DIR: ${WATCH_DIR}"
python3 /home/source_watcher.py ${WATCH_DIR} &

# run tornado
exec ${VIRTUAL_ENV}/bin/python /home/manage.py
