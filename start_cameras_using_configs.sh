#!/bin/bash -e
#######################################################################################################################
# This script creates streams from config file
# ./start_cameras_using_configs.sh 5 0 tests/results/camera_config_sky12_default
#######################################################################################################################
def_path="tests/results/camera_configs"
num_streams=$1
num_repeats="${2:-0}"
dir_of_configs="${3:-$def_path}"

pkill -9 -f start_cameras_using_configs.py || true
pkill -9 -f ffmpeg || true

if [ "$num_streams" == "" ]; then
    echo "Number of streams not specified: 1, 2, 5, 10, 50, 100"
else
    echo "Using configs in $dir_of_configs"
    echo "Running $num_streams streams ..."

    python tests/start_cameras_using_configs.py -c "${dir_of_configs}/camera_config_${num_streams}.yaml" -r ${num_repeats}
fi
