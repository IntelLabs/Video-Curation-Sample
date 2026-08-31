#!/bin/bash
set -e

echo "Sweeping stale shared memory segments from previous runs..."
# Delete only Python-specific shared memory blocks to avoid messing up other host processes
rm -f /dev/shm/psm_*

# Run the download script using container env vars
echo "Starting model download..."
python /home/include/models.py -o /var/www/cache/model_classes.json

# Execute the CMD (which will be /home/manage.sh)
exec "$@"