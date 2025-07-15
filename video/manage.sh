#!/bin/bash -e

# UDF server
# cd /home/remote_function/
# python3 udf_server.py 5011 . &
# sleep 10

# Watch directory
python3 /home/watch_and_send2vdms.py /var/www/mp4 &

# mv /var/*.mp4 /var/www/mp4/ || true
if [ "${IN_SOURCE}" = "videos" ]; then
    # echo "Preprocessing videos ..."
    # start_t=$SECONDS
    python3 /home/segment_archive.py &
    # elapsed_t=$((SECONDS - start_t))
    # echo "Preprocessing took $elapsed_t seconds"
fi

# Watch directory
# python3 /home/watch_and_send2vdms.py /var/www/mp4 &

# run tornado
exec /home/manage.py
