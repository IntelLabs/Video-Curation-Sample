#!/bin/bash
set -e

# Run the download script using container env vars
echo "Starting model download..."
FILE="/home/resources/models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
if [ ! -f "$FILE" ]; then
    omz_downloader --list /home/resources/models/models.lst -o /home/resources/models --precisions FP16
fi
python /home/resources/models/download_yolo.py

# Execute the CMD (which will be /home/manage.sh)
exec "$@"