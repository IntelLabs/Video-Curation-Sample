import json
import os
import time
import uuid

DEBUG = os.environ.get("DEBUG", "0")


""" MAIN FUNCTION """


def run(ipfilename, format, options, tmp_dir_path):
    METADATA = options["metadata"]  # json.loads(options["metadata"])
    W, H = options["input_sizeWH"]
    otype = options["otype"]
    metadata = dict(
        sorted(
            METADATA.items(), key=lambda item: int(item[0].split("_")[0]), reverse=False
        )
    )

    # print(f"[DEBUG UDF METADATA] {metadata}", flush=True)

    response = {"opFile": ipfilename, "metadata": metadata}

    jsonfile = "jsonfile" + uuid.uuid1().hex + ".json"
    with open(jsonfile, "w") as f:
        json.dump(response, f, indent=4)

    if DEBUG == "1":
        num_detections = len(metadata.keys())
        print(f"[TIMING],end_udf_metadata,{ipfilename}," + str(time.time()), flush=True)

        print(
            f"[METADATA_INFO],{ipfilename},{otype},{num_detections},{W},{H}", flush=True
        )

    return ipfilename, jsonfile
