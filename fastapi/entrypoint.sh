#!/bin/bash
set -e

# Run the download script using container env vars
echo "Starting model download..."
python /home/include/models.py

# Execute the CMD (which will be /home/manage.sh)
exec "$@"