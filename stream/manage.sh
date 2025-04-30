#!/bin/bash -e

# Watch dir for file changes
export WATCH_DIR=/var/www/mp4

# ingest stream
cd ${WATCH_DIR}

#FFMPEG options
packet_size=18800 #1358 #18800
time_segment_s=10

SEGMENT_OPTS="-segment_time ${time_segment_s} -f segment -use_wallclock_as_timestamps 1 -reset_timestamps 1 -strftime 1 ${WATCH_DIR}/${HOSTNAME}__%Y-%m-%d_%H-%M-%S.mp4"
GENERAL_OPTS="-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -threads 1 -c copy -flush_packets 0"
VIDEO_OPTS="-f mpegts -movflags faststart -crf 28 -r 15"  #-vcodec libx264   -s 640x360

completed="INCOMPLETE"
while true; do
    ffmpeg -i udp://127.0.0.1:8088?pkt_size=${packet_size} ${GENERAL_OPTS} ${VIDEO_OPTS} -force_key_frames "expr:gte(t,n_forced*${time_segment_s})" ${SEGMENT_OPTS}
    # SEND: ffmpeg -re -i <mp4 video> -c copy -f mpegts -flush_packets 0 "udp://<hostname>:<port>?pkt_size=18800"
done
