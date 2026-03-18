#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
PLATFORM="${1:-Xeon}"
NCURATIONS="$2"
INGESTION="$3"
IN_SOURCE="$4"
NCPU="$5"
REGISTRY="$6"
NSTREAMS="$7"
DEVICE="$8"
DEBUG="$9"
DOCKER_TAR="${10}"
DOCKER_TAR_DIR="${11}"
RESIZE_FLAG="${12}"
MODEL_NAME="${13}"
CUSTOM_MODEL_FLAG="${14}"
OMIT_DETECTIONS_FLAG="${15}"
HOSTIP=$(ip route get 8.8.8.8 | awk '/ src /{split(substr($0,index($0," src ")),f);print f[2];exit}')

echo "Generating templates with PLATFORM=${PLATFORM},NCURATIONS=${NCURATIONS},NSTREAMS=${NSTREAMS},INGESTION=${INGESTION},DEVICE=${DEVICE},IN_SOURCE=${IN_SOURCE},NCPU=${NCPU},HOSTIP=${HOSTIP},DEBUG=${DEBUG},DOCKER_TAR=${DOCKER_TAR},DOCKER_TAR_DIR=${DOCKER_TAR_DIR},RESIZE_FLAG=${RESIZE_FLAG},OMIT_DETECTIONS_FLAG=${OMIT_DETECTIONS_FLAG}"

BDIR=$(dirname $(dirname $DIR))
# echo "docker build --build-arg no_proxy --network host --file=${BDIR}/video/Dockerfile.base -t lcc_base_video_image:latest ${DIR}/video $(env | cut -f1 -d= | grep -E '_(proxy|REPO|VER)$' | sed 's/^/--build-arg /') --build-arg DEVICE=${DEVICE}"
# if video or fastapi in DIR; then
# if [[ "$DIR" == *video* || "$DIR" == *fastapi* ]]; then
#     echo "docker build --network host --file=${DIR}/../video/Dockerfile.base $@ -t lcc_base_video_image:latest ${DIR}/../video $(env | cut -f1 -d= | grep -E '_(proxy|REPO|VER)$' | sed 's/^/--build-arg /') --build-arg DEVICE=${DEVICE}"
docker build --build-arg DEVICE=${DEVICE} --network host --file="${BDIR}/video/Dockerfile.base" -t "lcc_base_video_image:latest" "${BDIR}/video" $(env | cut -f1 -d= | grep -E '_(proxy|REPO|VER)$' | sed 's/^/--build-arg /')
# fi

if test -f "${DIR}/docker-compose.yml.m4"; then
    echo "Generating docker-compose.yml"
    m4 -D${DEVICE} -Din_${IN_SOURCE} -DREGISTRY_PREFIX=$REGISTRY -DINGESTION="$INGESTION" -DDEVICE="$DEVICE" -DDEBUG="$DEBUG" -DNCURATIONS="${NCURATIONS}" -DHOSTIP="${HOSTIP}" -DNSTREAMS="${NSTREAMS}" -DIN_SOURCE="${IN_SOURCE}" -DNCPU="${NCPU}" -DRESIZE_FLAG="${RESIZE_FLAG}" -DMODEL_NAME="${MODEL_NAME}" -DCUSTOM_MODEL_FLAG="${CUSTOM_MODEL_FLAG}" -DOMIT_DETECTIONS_FLAG="${OMIT_DETECTIONS_FLAG}" -I "${DIR}" "${DIR}/docker-compose.yml.m4" > "${DIR}/docker-compose.yml"
fi

