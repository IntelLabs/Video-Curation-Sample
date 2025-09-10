#!/bin/bash -e
#######################################################################################################################
# This script runs the Curation application
#######################################################################################################################
# DEFAULT VARIABLES
INGESTION="object,face"
EXP_TYPE=compose
REGISTRY=None
NCPU=0
NCURATIONS=1
NSTREAMS=1
IN_SOURCE=stream
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
    "registry:"
    "resize"
    "model:"
    "ncurations:"
    "nstreams:"
    "ncpu:"
    "omit-det"
    "source:"
    "debug"
    "device:"
    "tars"
)

OPTS=$(getopt \
    --longoptions "$(printf "%s," "${LONG_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "hdlozi:t:r:m:n:v:c:s:e:" \
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
        -n or --ncurations  optional    Number of ingestion containers [Default: 1]
        -o or --omit-det    optional    By default, object detections are printed. To omit printing detections to screen, enable flag.
        -r or --registry    optional    Registry [Default: None]
        -s or --source      optional    Input source type (videos, stream) [Default: stream]
        -t or --type        optional    Deployment method (compose) [Default: compose]
        -v or --nstreams    optional    Number of video streams [Default: 1]
        -z or --resize      optional    Flag to resize video to model input size

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -c | --ncpu) shift; NCPU=$1; shift ;;
        -d | --debug) shift; DEBUG="1" ;;
        -l | --tars) shift; DOCKER_TAR="1" ;;
        -e | --device) shift; DEVICE=$1; shift ;;
        -i | --ingestion) shift; INGESTION=$1; shift ;;
        -m | --model) shift; MODEL_NAME=$1; shift ;;
        -n | --ncurations) shift; NCURATIONS=$1; shift ;;
        -r | --registry) shift; REGISTRY="$1"; shift ;;
        -s | --source) shift; IN_SOURCE="$1"; shift ;;
        -t | --type) shift; EXP_TYPE="$1"; shift ;;
        -v | --nstreams) shift; NSTREAMS=$1; shift ;;
        -z | --resize) shift; RESIZE_FLAG="True" ;;
        -o | --omit-det) shift; OMIT_DETECTIONS_FLAG="True" ;;
        --) shift; break ;;
        *) script_usage; exit 0 ;;
    esac
done

#######################################################################################################################
# BUILD AND START APP
cd $BUILD_DIR

if [ $REGISTRY == "None" ]; then
    cmake \
        -DDEBUG=$DEBUG \
        -DDEVICE=$DEVICE \
        -DDOCKER_TAR=$DOCKER_TAR \
        -DINGESTION=$INGESTION \
        -DIN_SOURCE=$IN_SOURCE \
        -DNCPU=$NCPU \
        -DNCURATIONS=$NCURATIONS \
        -DNSTREAMS=$NSTREAMS \
        -DRESIZE_FLAG=$RESIZE_FLAG \
        -DOMIT_DETECTIONS_FLAG=$OMIT_DETECTIONS_FLAG \
        -DMODEL_NAME=$MODEL_NAME \
        ..
else
    cmake \
        -DDEBUG=$DEBUG \
        -DDEVICE=$DEVICE \
        -DDOCKER_TAR=$DOCKER_TAR \
        -DINGESTION=$INGESTION \
        -DIN_SOURCE=$IN_SOURCE \
        -DNCPU=$NCPU \
        -DNCURATIONS=$NCURATIONS \
        -DNSTREAMS=$NSTREAMS \
        -DRESIZE_FLAG=$RESIZE_FLAG \
        -DOMIT_DETECTIONS_FLAG=$OMIT_DETECTIONS_FLAG \
        -DMODEL_NAME=$MODEL_NAME \
        -DREGISTRY=$REGISTRY \
        ..
fi

make

if [ $EXP_TYPE == "compose" ]; then
    make start_docker_compose

# elif [ $EXP_TYPE == "k8" ]; then
#     if [ $REGISTRY == "None" ]; then
#         make update
#     fi

#     make start_kubernetes

else
    echo "INVALID TYPE: ${EXP_TYPE}"

fi

cd $DIR
