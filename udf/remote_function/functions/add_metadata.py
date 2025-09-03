import os
from time import sleep, time

import vdms

DEBUG = os.environ.get("DEBUG", "0")
LOCKTIMEOUT_RETRIES = 5
ERR_KEYWORDS = ["timeout", "null search iterator", "outoftransactions"]


def retry_query(db, query, num_retries=LOCKTIMEOUT_RETRIES, sleep_timer: int = 0):
    # ridx = 0
    # while True:
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if "FailedCommand" in response[0] and any(
            k in response[0]["info"].lower() for k in ERR_KEYWORDS
        ):
            err = response[0]["info"]
            if DEBUG == "1":
                print(
                    f"[DEBUG add_metadata Attempt #{ridx}] Received '{err}' for {query} ",
                    flush=True,
                )
            if sleep_timer > 0:
                sleep(sleep_timer)
            # ridx += 1
            # pass  # Rerun
        else:
            if DEBUG == "1":
                print(
                    f"[DEBUG add_metadata] Successful query response: {response}",
                    flush=True,
                )
            break  # Continue
    return response


def run(ipfilename, format, options, tmp_dir_path):
    op_name = options["Name"]
    start_t = time()
    if DEBUG == "1":
        print(
            "[DEBUG] Adding metadata for {} from UDF to host: {} and port: {}".format(
                op_name, options["host"], options["port"]
            ),
            flush=True,
        )
        print(
            f"[TIMING],start_bkgd_add_metadata,{op_name},{start_t}",
            flush=True,
        )
    db = vdms.vdms()
    db.connect(options["host"], options["port"])

    if DEBUG == "1":
        print(
            "[DEBUG add_metadata] {} metadata keys: {}".format(
                op_name, list(options["metadata"].keys())
            ),
            flush=True,
        )

    ref = 1
    query = [
        {
            "FindVideo": {
                "_ref": ref,
                "constraints": {
                    "uid": ["==", options["uid"]],
                },
                "results": {"limit": 1},
            }
        }
    ]
    fref = 0
    for k in options["metadata"]:
        metadata = options["metadata"][k]
        fref += 2
        add_query = {
            "AddEntity": {
                "_ref": fref,
                "class": "Frame",
                "properties": metadata["frame_props"],
            }
        }

        add_frame_conn_query = {
            "AddConnection": {
                "class": "Vid2Frame",
                "properties": metadata["edge_props"],
                "ref1": 1,
                "ref2": fref,
            }
        }

        add_bbox_query = {
            "AddBoundingBox": {
                "_ref": fref + 1,
                "properties": metadata["bbox_props"],
                "rectangle": {
                    "h": int(metadata["bbox_props"]["VD:height"]),
                    "w": int(metadata["bbox_props"]["VD:width"]),
                    "x": int(metadata["bbox_props"]["VD:x1"]),
                    "y": int(metadata["bbox_props"]["VD:y1"]),
                },
            }
        }

        add_bbox_conn_query = {
            "AddConnection": {
                "class": "Frame2BB",
                "properties": metadata["bb_edge_props"],
                "ref1": fref,
                "ref2": fref + 1,
            }
        }

        query.append(add_query)
        query.append(add_frame_conn_query)
        query.append(add_bbox_query)
        query.append(add_bbox_conn_query)

    # response, res_arr = db.query(query, [[]])
    # print(response)
    response = retry_query(db, query, num_retries=10, sleep_timer=5)
    end_t = time()

    if DEBUG == "1":
        e_time = end_t - start_t
        print(
            f"[DEBUG] bkgd_add_metadata elapsed time: {e_time} secs",
            flush=True,
        )
        print(
            f"[TIMING],end_bkgd_add_metadata,{op_name},{end_t}",
            flush=True,
        )
        print(
            "[DEBUG] {} BACKGROUND ADD_METADATA RESPONSE: {}".format(op_name, response),
            flush=True,
        )

    return ipfilename, None
