#!/bin/bash -e

IMAGE="lcc_video"
DIR=$(dirname $(readlink -f "$0"))
IN_SOURCE="$4"

if [[ $IN_SOURCE == *"videos"* ]]; then
    "$DIR/download.sh"
fi
. "$DIR/../script/build.sh"
