#!/bin/bash -e

IMAGE="lcc_stream"
DIR=$(dirname $(readlink -f "$0"))

. "$DIR/../script/build.sh" 
