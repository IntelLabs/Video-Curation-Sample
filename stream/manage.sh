#!/bin/bash -e

# Watch dir for file changes
export WATCH_DIR=/var/www/mp4

# ingest stream
cd ${WATCH_DIR}

#FFMPEG options
packet_size=18800 #1358 #18800
time_segment_s=10
# SEGMENT_OPTS_V1="-f stream_segment -segment_format mpegts -segment_time ${time_segment_s} -segment_atclocktime 1 -reset_timestamps 1 ${WATCH_DIR}/stream1_%03d.mp4"
SEGMENT_OPTS="-segment_time ${time_segment_s} -f segment -use_wallclock_as_timestamps 1 -reset_timestamps 1 -strftime 1 ${WATCH_DIR}/%Y-%m-%d_%H-%M-%S.mp4" # stream1_%03d.mp4"2
GENERAL_OPTS="-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -threads 1 -c copy -flush_packets 0" # -flags -global_header
VIDEO_OPTS="-vcodec libx264 -f mpegts -movflags faststart -crf 28 -s 640x360 -r 15"

completed="INCOMPLETE"
while true; do
    if [ "$STREAM_URL" != " " ] && [ "$completed" == "INCOMPLETE" ]; then
        ffmpeg -i ${STREAM_URL}?pkt_size=${packet_size} ${GENERAL_OPTS} ${VIDEO_OPTS} -force_key_frames "expr:gte(t,n_forced*${time_segment_s})" ${SEGMENT_OPTS}
        if [[ $STREAM_URL == *"http"* ]]; then
            completed="DONE"
            echo "Work: $completed"
        fi
    fi
done
