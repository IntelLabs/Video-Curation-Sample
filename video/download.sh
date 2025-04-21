#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
mkdir -p "${DIR}/archive" "${DIR}/log/"

CLIPS=($(cat "${DIR}"/streamlist.txt))

# LOGFILE="${DIR}/log/video_storage_local_download_metrics.csv"
# echo "clip_name,type,Elapsed_time_s,File_size" | tee ${LOGFILE}

for clip in "${CLIPS[@]}"; do
    url=$(echo "$clip" | cut -f1 -d',')
    clip_name=$(echo "$clip" | cut -f2 -d',')
    clip_mp4="${clip_name/\.*/}.mp4"
    license=$(echo "$clip" | cut -f3 -d',')

    if test ! -f "$DIR/archive/$clip_mp4"; then
        if test "$reply" = ""; then
            printf "\n\n\nThe Library Curation sample requires a set of videos for curation. Please accept downloading dataset for library curation:\n\nDataset: $url\nLicense: $license\n\nThe terms and conditions of the data set license apply. Intel does not grant any rights to the data files.\n\n\nPlease type \"accept\" or anything else to skip the download.\n"
            read reply
        fi
        if test "$reply" = "accept"; then
            # start_time=$(date +%s.%3N)
            wget -U "XXX YYY" -O "$DIR/archive/$clip_mp4" "$url"
            # end_time=$(date +%s.%3N)
            # file_size="$(du -h $DIR/archive/$clip_mp4 | cut -f 1)"
            # total_time=$(echo "$end_time - $start_time" | bc);
            # echo "${clip_mp4},download video,${total_time},${file_size}" >> ${LOGFILE}
        else
            echo "Skipping..."
        fi
    fi
done

if test "$(find $DIR/archive -name '*.mp4' -print | wc -l)" -eq 0; then
    printf "\n\nNo clip is detected for library curation.\n\nYou can use your own video dataset. The database must be stored under $DIR/archive and must contain MP4 files.\n\n"
    exit -1
fi
