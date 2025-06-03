#!/usr/bin/env python3

import os
import shlex
import subprocess
import sys

time_segment_s = 10  # segment every 10 secs
time_segment_half = time_segment_s / 2  # forces a keyframe at t=5,10,15 seconds.

GENERAL_OPTS = "-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -threads 1 -filter:v fps=15 -flush_packets 0"
VIDEO_OPTS = "-f mpegts -movflags faststart"  # -vcodec libx264   -s 640x360


def main(watch_folder="/var/www/mp4"):
    for filename in os.listdir(watch_folder):
        if filename.endswith(".mp4"):  # and not filename.startswith("video"):
            filename_path = os.path.join(watch_folder, filename)
            file_part = filename[:-4]

            if filename.startswith("video"):
                new_filename = f"{file_part}_%d.mp4"
            else:
                new_filename = f"video_{file_part}_%d.mp4"
            SEGMENT_OPTS = f"-segment_time {time_segment_s} -f segment -use_wallclock_as_timestamps 1 -reset_timestamps 1 {watch_folder}/{new_filename}"

            cmd_str = f"ffmpeg -i {filename_path} {GENERAL_OPTS} {VIDEO_OPTS} -force_key_frames expr:gte(t,n_forced*{time_segment_s}) {SEGMENT_OPTS}"
            cmd_list = shlex.split(cmd_str)
            # print(" ".join(cmd_list), flush=True)

            ffmpeg_result = subprocess.run(cmd_list, capture_output=True, text=True)
            if ffmpeg_result.returncode > 0:
                print("Error:", ffmpeg_result.stderr)
            else:
                os.remove(filename_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
