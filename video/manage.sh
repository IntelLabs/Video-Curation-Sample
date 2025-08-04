#!/bin/bash -e

# UDF server
cd /home/remote_function/
python3 udf_server.py 5011 . &
sleep 10

# Watch directory
# python3 /home/watch_and_send2vdms.py /var/www/archive &  # Default
python3 /home/watch_and_send2vdms_cb.py /var/www/archive &  # UDF for metadata

# run tornado
exec /home/manage.py
