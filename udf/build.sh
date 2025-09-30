#!/bin/bash -e

IMAGE="lcc_udf"
DIR=$(dirname $(readlink -f "$0"))

. "$DIR/../script/build.sh"
