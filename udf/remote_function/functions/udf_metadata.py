import json
import os
import time
import uuid

DEBUG = os.environ.get("DEBUG", "0")


""" MAIN FUNCTION """


def _sort_dict_by_frame(in_dict):
    def _by_int(key):
        # return int(key.split("_")[0])
        return tuple(int(k) for k in key.split("_"))

    return dict(sorted(in_dict.items(), key=lambda x: _by_int(x[0])))


def run(ipfilename, format, options, tmp_dir_path):
    local_filename = options["filename"] if "filename" in options else ipfilename
    METADATA = options["metadata"]
    W, H = options["input_sizeWH"]
    otype = options["otype"]

    if DEBUG == "1":
        print(
            f"[TIMING],start_udf_metadata_{otype},{local_filename},{time.time()}",
            flush=True,
        )

    metadata = _sort_dict_by_frame(METADATA)

    # Metadata is sorted here
    keys = list(metadata.keys())
    print(
        f"[DEBUG UDF METADATA] Metadata ({otype}) keys for {local_filename}: {keys}",
        flush=True,
    )

    response = {"opFile": ipfilename, "metadata": metadata}

    jsonfile = "jsonfile" + uuid.uuid1().hex + ".json"
    with open(jsonfile, "w") as f:
        json.dump(response, f, indent=4)

    if DEBUG == "1":
        num_detections = len(metadata.keys())
        print(
            f"[TIMING],end_udf_metadata_{otype},{local_filename},{time.time()}",
            flush=True,
        )

        print(
            f"[METADATA_INFO],{local_filename},{otype},{num_detections},{W},{H}",
            flush=True,
        )

    return ipfilename, jsonfile
