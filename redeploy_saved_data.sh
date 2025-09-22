#!/bin/bash -e
#######################################################################################################################
# Re-deploy saved data
#######################################################################################################################
# DEFAULT VARIABLES
# in_vol=$1
# out_vol=$2

# docker volume create ${out_vol}

# docker run --rm -v ${in_vol}:/src -v ${out_vol}:/dest alpine ash -c "cp -a /src/. /dest/"

#######################################################################################################################

# Remove any videos
rm inputs/*.mp4

# Rename existing camera config and create blank one
mv inputs/camera_config.yaml inputs/camera_config.yaml.bak
touch inputs/camera_config.yaml

# Rename saved volumes to expected names
./deployment/DockerImageTars/rename_volume.sh lcc_app-content_saved lcc_app-content
./deployment/DockerImageTars/rename_volume.sh lcc_app-content_saved lcc_vdms-content

# Start APP
./start_app.sh -e GPU --tars


