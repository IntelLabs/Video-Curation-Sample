#!/bin/bash -e

# Watch dir for file changes
export WATCH_DIR=/var/www/streams

# ingest stream
cd ${WATCH_DIR}

#FFMPEG options
packet_size=18800 #1358 #18800
time_segment_s=10

SEGMENT_OPTS="-segment_time ${time_segment_s} -f segment -use_wallclock_as_timestamps 1 -reset_timestamps 1 -strftime 1 ${WATCH_DIR}/${HOSTNAME}__%Y-%m-%d_%H-%M-%S.mp4"
GENERAL_OPTS="-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -threads 1 -filter:v fps=15 -flush_packets 0"
VIDEO_OPTS="-f mpegts -movflags faststart"  # -vcodec libx264 -s 640x360 -crf 28 -r 15

completed="INCOMPLETE"
time_segment_ns=$((time_segment_s * 1000000000))
timeout_value_mins=2
timeout_microsecs=$((timeout_value_mins * 60000000))
while true; do
    ffmpeg -timeout ${timeout_microsecs} -i udp://127.0.0.1:8088?pkt_size=${packet_size} ${GENERAL_OPTS} ${VIDEO_OPTS} -force_key_frames "expr:gte(t,n_forced*${time_segment_s})" ${SEGMENT_OPTS}
    # gst-launch-1.0 -v -e  udpsrc port=8088 ! video/mpegts, systemstream=true, clock-rate=90000 ! tsdemux ! queue ! h264parse ! splitmuxsink location=${HOSTNAME}__clip_%04d.mp4 max-size-time=${time_segment_ns} send-keyframe-requests=true
    # SEND: ffmpeg -re -i <mp4 video> -c copy -f mpegts -flush_packets 0 "udp://<hostname>:<port>?pkt_size=18800"
done
