#!/bin/bash -e
#######################################################################################################################
# Save docker images
#######################################################################################################################
# DEFAULT VARIABLES
RESIZE_FLAG="False"
DEVICE="CPU"

DIR=$(dirname $(readlink -f "$0"))

#######################################################################################################################
# GET SCRIPT OPTIONS

LONG_LIST=(
    "resize"
    "device:"
)

OPTS=$(getopt \
    --longoptions "$(printf "%s," "${LONG_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "hze:" \
    -- "$@"
)

eval set -- $OPTS

script_usage()
{
    cat <<EOF
    This script runs the Video Curation Streaming Application

    Usage: $0 [ options ]

    Options:
        -h                  optional    Print this help message
        -e or --device      optional    Device for inference (CPU, GPU) [Default: CPU]
        -z or --resize      optional    Flag to resize video to model input size

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -e | --device) shift; DEVICE=$1; shift ;;
        -z | --resize) shift; RESIZE_FLAG="True" ;;
        --) shift; break ;;
        *) script_usage; exit 0 ;;
    esac
done


if [ $RESIZE_FLAG == "True" ]; then
    SAVE_DIR="${DIR}/resize"
else
    SAVE_DIR="${DIR}/full"
fi

mkdir -p ${SAVE_DIR}

#######################################################################################################################
# SAVE IMAGES

echo "Saving lcc_frontend:stream ..."
docker save -o ${SAVE_DIR}/lcc_frontend_stream.tar lcc_frontend:stream

echo "Saving lcc_video:stream ..."
docker save -o ${SAVE_DIR}/lcc_video_stream.${DEVICE}.tar lcc_video:stream

echo "Saving lcc_vdms:stream ..."
docker save -o ${SAVE_DIR}/lcc_vdms_stream.tar lcc_vdms:stream

echo "Saving lcc_udf:stream ..."
docker save -o ${SAVE_DIR}/lcc_udf_stream.tar lcc_udf:stream

echo "Saving lcc_certificate:stream ..."
docker save -o ${SAVE_DIR}/lcc_certificate_stream.tar lcc_certificate:stream

echo "CONTAINER LOCATION: ${SAVE_DIR}"
