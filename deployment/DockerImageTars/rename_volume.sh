#!/bin/bash -e
#######################################################################################################################
# Rename volumes for service for later redeployment
#######################################################################################################################
# DEFAULT VARIABLES
in_vol=$1
out_vol=$2

docker volume create ${out_vol}

docker run --rm -v ${in_vol}:/src -v ${out_vol}:/dest alpine ash -c "cp -a /src/. /dest/"

