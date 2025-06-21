#!/bin/bash -e

echo "${INGEST_METHOD}"

if [ ${INGEST_METHOD} == "udf" ]; then
    # UDF server
    cd /home/remote_function/
    python3 udf_server.py 5011 . &
    sleep 10
fi

# Watch directory
python3 /home/watch_and_send2vdms.py /var/www/streams &

# run tornado
exec /home/manage.py
