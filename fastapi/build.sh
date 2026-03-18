#!/bin/bash -e

IMAGE="lcc_fastapi"
DIR=$(dirname $(readlink -f "$0"))

# Make sure user provided models are available to FastAPI
cp -rp "$DIR/../video/resources" $DIR/

. "$DIR/../script/build.sh"

rm -rf "$DIR/resources"
