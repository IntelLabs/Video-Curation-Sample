import os
from time import sleep, time

import vdms

DEBUG = os.environ.get("DEBUG", "0")
LOCKTIMEOUT_RETRIES = 5
ERR_KEYWORDS = [
    "timeout",
    "null search iterator",
    "outoftransactions",
    "no entities found",
]


def retry_query(db, query, num_retries=LOCKTIMEOUT_RETRIES, sleep_timer: int = 0):
    # ridx = 0
    # while True:
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if (
            "FailedCommand" in response[0]
            and any(k in response[0]["info"].lower() for k in ERR_KEYWORDS)
        ) or (
            ("FindVideo" in response[0])
            and ("info" in response[0]["FindVideo"])
            and any(k in response[0]["FindVideo"]["info"].lower() for k in ERR_KEYWORDS)
        ):
            if "FailedCommand" in response[0]:
                err = response[0]["info"]
            else:
                err = response[0]["FindVideo"]["info"]
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
                # "_ref": ref,
                "constraints": {
                    "uid": ["==", options["uid"]],
                },
                "results": {"limit": 1},
            }
        }
    ]

    # Make sure video exists first
    # If not (after few retries), skip adding BBs and move on
    # If so, continue processing BBs
    response = retry_query(db, query, sleep_timer=3)

    if (
        "FailedCommand" in response[0]
        and any(k in response[0]["info"].lower() for k in ERR_KEYWORDS)
    ) or (
        ("FindVideo" in response[0])
        and ("info" in response[0]["FindVideo"])
        and any(k in response[0]["FindVideo"]["info"].lower() for k in ERR_KEYWORDS)
    ):
        if DEBUG == "1":
            print(
                f"[DEBUG add_metadata] FindVideo failed for {op_name}. No BBs inserted for chunk",
                flush=True,
            )

    else:
        query[0]["FindVideo"]["_ref"] = ref
        fref = 2
        added_frames = {}
        for _, metadata in options["metadata"].items():
            # metadata = options["metadata"][k]
            # frameidx, framebbidx = frameidx_framebbidx.split("_")
            # fref += 2
            if metadata["frame_props"]["frameID"] not in added_frames:
                add_query = {
                    "AddEntity": {
                        "_ref": fref,
                        "class": "Frame",
                        "properties": metadata["frame_props"],
                        "constraints": {
                            "server_filepath": ["==", op_name],
                            "frameID": ["==", metadata["frame_props"]["frameID"]],
                        },
                    }
                }
                query.append(add_query)
                added_frames[metadata["frame_props"]["frameID"]] = fref
                fref += 1

            add_frame_conn_query = {
                "AddConnection": {
                    "class": "Vid2Frame",
                    "properties": metadata["edge_props"],
                    "ref1": 1,
                    "ref2": added_frames[metadata["frame_props"]["frameID"]],
                }
            }
            query.append(add_frame_conn_query)

            add_bbox_query = {
                "AddBoundingBox": {
                    "_ref": fref,  #  + 1,
                    "properties": metadata["bbox_props"],
                    "rectangle": {
                        "h": int(metadata["bbox_props"]["VD:height"]),
                        "w": int(metadata["bbox_props"]["VD:width"]),
                        "x": int(metadata["bbox_props"]["VD:x1"]),
                        "y": int(metadata["bbox_props"]["VD:y1"]),
                    },
                }
            }
            query.append(add_bbox_query)

            add_bbox_conn_query = {
                "AddConnection": {
                    "class": "Frame2BB",
                    "properties": metadata["bb_edge_props"],
                    "ref1": added_frames[metadata["frame_props"]["frameID"]],
                    "ref2": fref,  #  + 1,
                }
            }

            query.append(add_bbox_conn_query)
            fref += 1

        # response, res_arr = db.query(query, [[]])
        # print(response)
        response = retry_query(db, query, sleep_timer=3)

        if DEBUG == "1":
            print(
                "[DEBUG] {} BACKGROUND ADD_METADATA RESPONSE: {}".format(
                    op_name, response
                ),
                flush=True,
            )

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

    return ipfilename, None
