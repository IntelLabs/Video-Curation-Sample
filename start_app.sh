#!/bin/bash -e
#######################################################################################################################
# This script runs the Curation application
#######################################################################################################################
# DEFAULT VARIABLES
INGESTION="object"  #,face"
EXP_TYPE=compose
DEBUG="0"
DEVICE="CPU"
DOCKER_TAR="0"
RESIZE_FLAG="False"
OMIT_DETECTIONS_FLAG="False"
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
        -e or --device      optional    Device for inference (CPU, GPU) [Default: CPU]
        -i or --ingestion   optional    Ingestion type (object, face) [Default: "object,face"]
        -l or --tars        optional    Flag to load docker images instead of building from Dockerfiles
        -m or --model       optional    Custom YOLO model name (<model name>.pt). If not provided model YOLO11n is used.
        -o or --omit-det    optional    By default, object detections are printed. To omit printing detections to screen, enable flag.
        -t or --type        optional    Deployment method (compose) [Default: compose]
        -z or --resize      optional    Flag to resize video to model input size

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -d | --debug) shift; DEBUG="1" ;;
        -l | --tars) shift; DOCKER_TAR="1" ;;
        -e | --device) shift; DEVICE=$1; shift ;;
        -i | --ingestion) shift; INGESTION=$1; shift ;;
        -m | --model) shift; MODEL_NAME=$1; shift ;;
        -t | --type) shift; EXP_TYPE="$1"; shift ;;
        -z | --resize) shift; RESIZE_FLAG="True" ;;
        -o | --omit-det) shift; OMIT_DETECTIONS_FLAG="True" ;;
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
