#!/bin/bash
set -e

# Run the download script using container env vars
echo "Starting model download..."
python /home/include/models.py -o /var/www/cache/model_classes.json

# Execute the CMD (which will be /home/manage.sh)
exec "$@"