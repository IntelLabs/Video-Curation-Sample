#!/bin/bash -e
#######################################################################################################################
# This script runs the Curation application
#######################################################################################################################
# DEFAULT VARIABLES
INGESTION="object"  #,face"
EXP_TYPE=compose
DEBUG="0"
DEVICE="GPU"
DOCKER_TAR="0"
RESIZE_FLAG="False"
OMIT_DETECTIONS_FLAG="True"
MODEL_NAME=""

DIR=$(dirname $(readlink -f "$0"))
BUILD_DIR=$DIR/build

LONG_LIST=(
    "ingestion:"
    "type:"
    "resize"
    "model:"
    "omit-det"
    "debug"
    "device:"
    "tars"
)

OPTS=$(getopt \
    --longoptions "$(printf "%s," "${LONG_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "hdlozi:t:m:e:" \
    -- "$@"
)

eval set -- $OPTS

if [ -d "$BUILD_DIR" ]; then
    rm -rf $BUILD_DIR
fi

mkdir -p $BUILD_DIR

#######################################################################################################################
# GET SCRIPT OPTIONS
script_usage()
{
    cat <<EOF
    This script runs the Video Curation Streaming Application

    Usage: $0 [ options ]

    Options:
        -h                  optional    Print this help message
        -d or --debug       optional    Flag to enable debug messages
        -l or --tars        optional    Flag to load docker images instead of building from Dockerfiles
        -m or --model       optional    Custom YOLO model name (<model name>.pt). If not provided model YOLO11n is used.

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -d | --debug) shift; DEBUG="1" ;;
        -l | --tars) shift; DOCKER_TAR="1" ;;
        -m | --model) shift; MODEL_NAME=$1; shift ;;
        --) shift; break ;;
        *) script_usage; exit 0 ;;
    esac
done

#######################################################################################################################
# BUILD AND START APP
cd $BUILD_DIR

cmake \
    -DDEBUG=$DEBUG \
    -DDEVICE=$DEVICE \
    -DDOCKER_TAR=$DOCKER_TAR \
    -DINGESTION=$INGESTION \
    -DMODEL_NAME=$MODEL_NAME \
    -DOMIT_DETECTIONS_FLAG=$OMIT_DETECTIONS_FLAG \
    -DRESIZE_FLAG=$RESIZE_FLAG \
    ..

make

if [ $EXP_TYPE == "compose" ]; then
    make start_docker_compose

else
    echo "INVALID TYPE: ${EXP_TYPE}"

fi

cd $DIR
