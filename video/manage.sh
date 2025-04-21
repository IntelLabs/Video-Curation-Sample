#!/bin/bash -e

# Watch directory
# ./watch-new-clips.sh &
python3 /home/watch_and_notify.py /var/www/streams &

# run tornado
exec /home/manage.py
