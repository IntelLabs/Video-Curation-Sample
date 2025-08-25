import argparse
import re
import time
from math import ceil
from pathlib import Path

import pandas as pd
import yaml


def secs2HMS_str(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60

    time_str = ""
    if hour > 0:
        time_str += f"{hour} hrs"
    if min > 0:
        time_str += f" {min} mins"
    if sec > 0:
        time_str += f" {sec:04f} secs"
    return time_str


def read_config(file_path):
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


PROJECT_PATH = Path(__file__).parent.parent.absolute()
clip_length_in_secs = 10
TARGET_FPS = 15
KEYWORDS = [
    "[TIMING]",
    "[METADATA_INFO]",
    # "e2e_query_processing",
    "[DEBUG]",
    # "OBJECT DETECTION]",
    # "METADATA]",
    "Clip,contains,frames",
    "struct.error: unpack requires a buffer of 4 bytes",
]

PROCESSING_DEFAULT_DICT = {
    "num clips": 0,
    "total clip frames": 0,  # info_details["Num Frames"]
    "frameW": 0,
    "frameH": 0,
    "object detections": 0,
    "face detections": 0,
    "Num Failures": 0,
    "Num Bkgd Failures": 0,
    "Time to create clip (s)": 0,  # info_details["Save clip"] - info_details["Start new clip"]
    "UDF object db.query runtime (s)": 0,
    "UDF object run func runtime (s)": 0,
    "UDF face db.query runtime (s)": 0,
    "UDF face run func runtime (s)": 0,
}


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dir",
        dest="log_dir",
        default=PROJECT_PATH / "tests/results/camera_configs",
        type=Path,
        help="Directory containing log from app run with extension '.log'",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search recursively for camera_config_*.log files",
    )

    args = parser.parse_args()
    args.log_dir = args.log_dir.resolve()
    args.csv_file = str(args.log_dir / "log_summary.csv")
    txt_file = str(args.log_dir / "log_summary.txt")
    args.out_log_file = open(txt_file, "w")
    return args


def get_overall_details(info_details, out_log_file):
    app_info = {}
    VDMS_crashes = 0
    if "start_watchandsend" not in info_details:
        app_end_time = info_details["Min Timestamp"]
    else:
        app_end_time = info_details["start_watchandsend"]

    if "end_watchandsend" not in info_details:
        app_end_time = info_details["Max Timestamp"]
    else:
        app_end_time = info_details["end_watchandsend"]

    watch_process_elapsed_time = app_end_time - info_details["start_watchandsend"]
    time_str = secs2HMS_str(watch_process_elapsed_time)
    print(
        f"\t[Overall] App took {time_str} to process all videos/streams",
        flush=True,
        file=out_log_file,
    )
    app_info["App processing time (s)"] = watch_process_elapsed_time

    stream_process_elapsed_time = app_end_time - info_details["Min Timestamp"]
    time_str = secs2HMS_str(stream_process_elapsed_time)
    print(
        f"\t[Overall] Took {time_str} from stream start to all videos/streams processed\n",
        flush=True,
        file=out_log_file,
    )
    app_info["Stream processing time (s)"] = stream_process_elapsed_time

    if "VDMS crashes" in info_details:
        VDMS_crashes = info_details["VDMS crashes"]

    app_info["Log VDMS crashes"] = VDMS_crashes

    return app_info


def summarize_info(log_filename, info, out_log_file=None, method=None):
    app_info = {}
    camera_info = {}

    for key, info_details in info.items():
        if key == "Overall":
            # VDMS_crashes = 0
            # if "start_watchandsend" not in info_details:
            #     app_end_time = info_details["Min Timestamp"]
            # else:
            #     app_end_time = info_details["start_watchandsend"]

            # if "end_watchandsend" not in info_details:
            #     app_end_time = info_details["Max Timestamp"]
            # else:
            #     app_end_time = info_details["end_watchandsend"]

            # watch_process_elapsed_time = (
            #     app_end_time - info_details["start_watchandsend"]
            # )
            # time_str = secs2HMS_str(watch_process_elapsed_time)
            # print(
            #     f"\t[Overall] App took {time_str} to process all videos/streams",
            #     flush=True,
            #     file=out_log_file,
            # )
            # app_info["App processing time (s)"] = watch_process_elapsed_time

            # stream_process_elapsed_time = app_end_time - info_details["Min Timestamp"]
            # time_str = secs2HMS_str(stream_process_elapsed_time)
            # print(
            #     f"\t[Overall] Took {time_str} from stream start to all videos/streams processed\n",
            #     flush=True,
            #     file=out_log_file,
            # )
            # app_info["Stream processing time (s)"] = stream_process_elapsed_time

            # if "VDMS crashes" in info_details:
            #     VDMS_crashes = info_details["VDMS crashes"]

            # app_info["Log VDMS crashes"] = VDMS_crashes

            app_info.update(get_overall_details(info_details, out_log_file))

        elif ".mp4" not in key:
            camera_info.setdefault(key, {})
            frames_received = 0
            received_expected_clips = 0
            frames_processed = 0
            stream_processing_time = 0
            delta_streamend_2_processing_end = 0

            video_name = info_details["streamed video"]
            camera_info[key]["video"] = video_name
            camera_info[key]["video duration (s)"] = info_details["video duration (s)"]
            camera_info[key]["video fps"] = info_details["video fps"]
            camera_info[key]["video frames"] = int(
                info_details["video fps"] * info_details["video duration (s)"]
            )
            camera_info[key]["video expected clips"] = ceil(
                info_details["video duration (s)"] / clip_length_in_secs
            )
            new_target_fps = (
                TARGET_FPS
                if info_details["video fps"] > TARGET_FPS
                else info_details["video fps"]
            )
            camera_info[key]["target fps"] = new_target_fps
            camera_info[key]["target frames"] = int(
                new_target_fps * info_details["video duration (s)"]
            )
            camera_info[key]["target expected clips"] = ceil(
                camera_info[key]["target frames"]
                / (clip_length_in_secs * new_target_fps)
            )

            if "frames received" in info_details:
                frames_received = info_details["frames received"]
                received_expected_clips = ceil(
                    (frames_received / info_details["video fps"]) / clip_length_in_secs
                )

            if "frames processed" in info_details:
                frames_processed = info_details["frames processed"]

            camera_info[key]["frames received"] = frames_received
            camera_info[key]["received expected clips"] = received_expected_clips
            camera_info[key]["frames processed"] = frames_processed
            camera_info[key]["stream send elapsed time (s)"] = info_details[
                "stream elapsed time (s)"
            ]
            time_str = secs2HMS_str(camera_info[key]["stream send elapsed time (s)"])
            print(
                f"\t[{key}] Took {time_str} to send {video_name}",
                flush=True,
                file=out_log_file,
            )

            if "Completed processing" in info_details:
                stream_processing_time = (
                    info_details["Completed processing"]
                    - info_details["Start processing"]
                )
                time_str = secs2HMS_str(stream_processing_time)
                print(
                    f"\t[{key}] Took {time_str} to process {video_name}",
                    flush=True,
                    file=out_log_file,
                )

                delta_streamend_2_processing_end = abs(
                    info_details["Completed processing"]
                    - info_details["stream end time"]
                )
                time_str = secs2HMS_str(delta_streamend_2_processing_end)
                print(
                    f"\t[{key}] Took {time_str} after stream ended to complete processing {video_name}",
                    flush=True,
                    file=out_log_file,
                )

            camera_info[key]["stream processing time (s)"] = stream_processing_time
            camera_info[key]["delta stream end to processing end (s)"] = (
                delta_streamend_2_processing_end
            )

            camera_info[key]["delta stream start to processing start (s)"] = abs(
                info_details["Start processing"] - info_details["stream start time"]
            )
            time_str = secs2HMS_str(
                camera_info[key]["delta stream start to processing start (s)"]
            )
            print(
                f"\t[{key}] Took {time_str} after starting stream to start processing {video_name}\n",
                flush=True,
                file=out_log_file,
            )

        elif ".mp4" in key:
            camera_name = key.split("_")[0]
            if "num clips" not in camera_info[camera_name]:
                camera_info[camera_name].update(PROCESSING_DEFAULT_DICT)

            camera_info[camera_name]["num clips"] += 1

            if "Num Frames" in info_details:
                camera_info[camera_name]["total clip frames"] += info_details[
                    "Num Frames"
                ]

            if "frameW" in info_details:
                camera_info[camera_name]["frameW"] = info_details["frameW"]
                camera_info[camera_name]["frameH"] = info_details["frameH"]

            if "object detections" in info_details:
                camera_info[camera_name]["object detections"] += info_details[
                    "object detections"
                ]

            if "face detections" in info_details:
                camera_info[camera_name]["face detections"] += info_details[
                    "face detections"
                ]

            if "Num Failures" in info_details:
                camera_info[camera_name]["Num Failures"] += info_details["Num Failures"]

            if "Num Bkgd Failures" in info_details:
                camera_info[camera_name]["Num Bkgd Failures"] += info_details[
                    "Num Bkgd Failures"
                ]

            if "Save clip" in info_details:
                camera_info[camera_name]["Time to create clip (s)"] += (
                    info_details["Save clip"] - info_details["Start new clip"]
                )

            if "end_udf_ingest_object" in info_details:
                camera_info[camera_name]["UDF object db.query runtime (s)"] += (
                    info_details["end_udf_ingest_object"]
                    - info_details["start_udf_ingest_object"]
                )
            # camera_info[camera_name]["VDMS object e2e runtime (s)"] += info_details["object e2e_query_processing (s)"]

            if "end_udf_metadata_object" in info_details:
                camera_info[camera_name]["UDF object run func runtime (s)"] += (
                    info_details["end_udf_metadata_object"]
                    - info_details["start_udf_metadata_object"]
                )

            if "end_udf_ingest_face" in info_details:
                camera_info[camera_name]["UDF face db.query runtime (s)"] += (
                    info_details["end_udf_ingest_face"]
                    - info_details["start_udf_ingest_face"]
                )
            # camera_info[camera_name]["VDMS face e2e runtime (s)"] += info_details["face e2e_query_processing (s)"]
            if "end_udf_metadata_face" in info_details:
                camera_info[camera_name]["UDF face run func runtime (s)"] += (
                    info_details["end_udf_metadata_face"]
                    - info_details["start_udf_metadata_face"]
                )

    details = []
    for name, cam_details in camera_info.items():
        cam_dict = {
            "log": log_filename,
            "Method": method,
            "stream name": name,
            "Log VDMS crashes": int(app_info["Log VDMS crashes"]),
        }

        if "num clips" not in cam_dict:
            cam_dict.update(PROCESSING_DEFAULT_DICT)

        # if "Log VDMS crashes" in app_info:
        #     cam_dict["Log VDMS crashes"] += int(app_info["Log VDMS crashes"])

        for k, v in cam_details.items():
            cam_dict[k] = v

        details.append(cam_dict)

    new_df = pd.DataFrame(details)

    # Reorder
    new_col_order = [
        "log",
        "Method",
        "stream name",
        "video",
        "video duration (s)",
        "video fps",
        "video frames",
        "video expected clips",
        "target fps",
        "target frames",
        "target expected clips",
        "received expected clips",
        "frames received",
        "frames processed",
        "num clips",
        "total clip frames",
        "frameW",
        "frameH",
        "object detections",
        "face detections",
        "Num Failures",
        "Num Bkgd Failures",
        "Log VDMS crashes",
        "stream send elapsed time (s)",
        "stream processing time (s)",
        "delta stream start to processing start (s)",
        "delta stream end to processing end (s)",
        "Time to create clip (s)",
        "UDF object db.query runtime (s)",
        "UDF object run func runtime (s)",
        "UDF face db.query runtime (s)",
        "UDF face run func runtime (s)",
    ]
    new_df = new_df[new_col_order]
    return new_df


def remove_value_from_list(the_list, value):
    if value in the_list:
        the_list.remove(value)
    return the_list


def get_log_info(args, log_path, method=None):  # Extract timing from logs
    min_timestamp = time.time()
    max_timestamp = 0
    camera_details = {}
    lines = []
    info = {"Overall": {}}
    mp4_pattern = r"\b(\S+\.mp4)\b"
    # meta_mp4_file = None
    # meta_ingest_type = None
    with open(log_path, "r") as log:
        file_desc = f"{log_path.name}"
        if method is not None:
            file_desc += f" ({method})"
        print(f"\n[{file_desc}]", flush=True, file=args.out_log_file)

        # Get streaming details
        stream_video_results = str(log_path).replace(".log", ".videos.yaml")
        if Path(stream_video_results).exists():
            camera_details = read_config(stream_video_results)
            for camera_name, details in camera_details.items():
                info.setdefault(camera_name, {})

                if camera_name in info:  # Interested in those captured in logs
                    # info.setdefault(camera_name, {})
                    info[camera_name]["type"] = details["type"]
                    info[camera_name]["url"] = details["url"]
                    info[camera_name]["streamed video"] = details["video"].split("/")[
                        -1
                    ]
                    info[camera_name]["video duration (s)"] = details["duration"]
                    info[camera_name]["video fps"] = details["FPS"]
                    info[camera_name]["stream start time"] = details["start_time"]
                    info[camera_name]["stream end time"] = details["end_time"]
                    info[camera_name]["stream elapsed time (s)"] = details[
                        "elapsed_time_s"
                    ]
                    min_timestamp = min(min_timestamp, float(details["start_time"]))
                    max_timestamp = max(max_timestamp, float(details["end_time"]))
        else:
            return {}
        # Get processing details
        del_camera_names = list(camera_details.keys())
        for line in log:
            if any(all(k in line for k in kl.split(",")) for kl in KEYWORDS):
                line = line.replace("\n", "")

                files = [
                    re.search(mp4_pattern, line_part).group(1)
                    for line_part in line.split(",")
                    if re.search(mp4_pattern, line_part)
                    and "db/" not in re.search(mp4_pattern, line_part).group(1)
                ]
                if len(files) != 0:
                    file = files[0]
                    info.setdefault(file, {})

                    frame_log_available = all(
                        sub in line for sub in ["[DEBUG]", "contains ", " frames"]
                    )
                    if frame_log_available:
                        info[file]["Num Frames"] = int(
                            line.split("contains ")[-1].split(" frames")[0]
                        )

                    elif "[DEBUG]" in line and not frame_log_available:
                        # if "PROPERTIES:" in line:
                        #     props = json.loads(line.split("PROPERTIES: ")[1].replace("'", '"'))

                        if "INGEST_VIDEO RESPONSE:" in line:
                            response = eval(line.split("INGEST_VIDEO RESPONSE: ")[1])
                            info[file].setdefault("Num Failures", 0)
                            if ("FailedCommand" in response[0]) or (
                                "AddVideo" in response[0]
                                and response[0]["AddVideo"]["status"] != 0
                            ):
                                info[file]["Num Failures"] += 1

                        if "BACKGROUND ADD_METADATA RESPONSE:" in line:
                            response = eval(
                                line.split("BACKGROUND ADD_METADATA RESPONSE: ")[1]
                            )
                            info[file].setdefault("Num Bkgd Failures", 0)
                            if "FailedCommand" in response[0]:
                                info[file]["Num Bkgd Failures"] += 1

                    elif "[METADATA_INFO]" in line:
                        (
                            prefix,
                            mp4_file,
                            ingest_type,
                            num_detections,
                            frameW,
                            frameH,
                        ) = line.split("|")[-1].strip().split(",")
                        info[mp4_file]["frameW"] = int(frameW)
                        info[mp4_file]["frameH"] = int(frameH)
                        info[mp4_file][f"{ingest_type} detections"] = int(
                            num_detections
                        )
                        # meta_mp4_file = mp4_file
                        # meta_ingest_type = ingest_type

                    elif "[TIMING]" in line:
                        prefix, method_name, name_or_mp4_file, timestamp = (
                            line.split("|")[-1].strip().split(",")
                        )
                        info[name_or_mp4_file][method_name] = float(timestamp)
                        min_timestamp = min(min_timestamp, float(timestamp))
                        max_timestamp = max(max_timestamp, float(timestamp))

                    else:
                        lines.append(line)

                # elif "e2e_query_processing" in line and meta_mp4_file is not None and meta_ingest_type is not None:
                #     e2etime = float(line.split(":")[-1]) / 1e6  # microsec to sec
                #     info[meta_mp4_file].setdefault(f"{meta_ingest_type} e2e_query_processing (s)", 0)
                #     info[meta_mp4_file][f"{meta_ingest_type} e2e_query_processing (s)"] += e2etime
                #     meta_mp4_file = None
                #     meta_ingest_type = None

                elif "struct.error: unpack requires a buffer of 4 bytes" in line:
                    info["Overall"].setdefault("VDMS crashes", 0)
                    info["Overall"]["VDMS crashes"] += 1

                elif "_watchandsend" in line:
                    prefix, method_name, _, timestamp = (
                        line.split("|")[-1].strip().split(",")
                    )
                    info["Overall"][method_name] = float(timestamp)
                    # min_timestamp = min(min_timestamp, float(timestamp))
                    max_timestamp = max(max_timestamp, float(timestamp))

                elif any(
                    sub in line for sub in ["Start processing", "Completed processing"]
                ):
                    prefix, method_name, stream_name, timestamp = (
                        line.split("|")[-1].strip().split(",")
                    )
                    info.setdefault(stream_name, {})
                    info[stream_name][method_name] = float(timestamp)
                    min_timestamp = min(min_timestamp, float(timestamp))
                    max_timestamp = max(max_timestamp, float(timestamp))
                    del_camera_names = remove_value_from_list(
                        del_camera_names, stream_name
                    )

                elif all(
                    sub in line
                    for sub in ["[DEBUG]", "Stream name:", "Num. Retrieved Frames"]
                ):
                    list_of_info = line.split("[DEBUG] ")[-1].split(", ")
                    stream_name = list_of_info[0].split("Stream name:")[-1].strip()
                    processor_time = float(
                        list_of_info[2].split("Elapsed Time:")[-1].strip()
                    )
                    num_received = int(
                        list_of_info[3].split("Num. Retrieved Frames:")[-1].strip()
                    )
                    num_processed = int(
                        list_of_info[4].split("Num. Processed Frames:")[-1].strip()
                    )

                    info[stream_name]["processor time (s)"] = processor_time
                    info[stream_name]["frames received"] = num_received
                    info[stream_name]["frames processed"] = num_processed
                    del_camera_names = remove_value_from_list(
                        del_camera_names, stream_name
                    )

                else:
                    lines.append(line)

                # line_idx += 1
        for stream_name in del_camera_names:
            if len(info[stream_name].keys()) == 8:
                del info[stream_name]

        info["Overall"]["Min Timestamp"] = min_timestamp
        info["Overall"]["Max Timestamp"] = max_timestamp

        info = dict(sorted(info.items(), key=lambda item: item[0], reverse=False))

    # print(lines)
    return info


def main(args):
    df = pd.DataFrame()

    if args.recursive:
        glob_cmd = args.log_dir.rglob("camera_config_*.log")
    else:
        glob_cmd = args.log_dir.glob("camera_config_*.log")

    for log_path in glob_cmd:
        method = None
        if args.recursive:
            method = log_path.parent.name

        # Extract timing from logs
        info = get_log_info(args, log_path, method=method)

        if info != {}:
            # Summarize info
            new_df = summarize_info(
                log_path.name, info, out_log_file=args.out_log_file, method=method
            )

            # Accumulate results
            df = pd.concat([df, new_df], ignore_index=True)

    # Write to file
    df.to_csv(args.csv_file, index=False)


if __name__ == "__main__":
    in_params = get_input_args()
    main(in_params)

    in_params.out_log_file.close()
