#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))

CLIPS=($(cat "${DIR}"/streamlist.txt))

for clip in "${CLIPS[@]}"; do
    url=$(echo "$clip" | cut -f1 -d',')
    clip_name=$(echo "$clip" | cut -f2 -d',')
    clip_mp4="${clip_name/\.*/}.mp4"
    license=$(echo "$clip" | cut -f3 -d',')

    if test ! -f "$DIR/$clip_mp4"; then
        if test "$reply" = ""; then
            printf "\n\n\nThe Library Curation sample requires a set of videos for curation. Please accept downloading dataset for library curation:\n\nDataset: $url\nLicense: $license\n\nThe terms and conditions of the data set license apply. Intel does not grant any rights to the data files.\n\n\nPlease type \"accept\" or anything else to skip the download.\n"
            read reply
        fi
        if test "$reply" = "accept"; then
            wget -U "XXX YYY" -O "$DIR/$clip_mp4" "$url"
        else
            echo "Skipping..."
        fi
    fi
done
