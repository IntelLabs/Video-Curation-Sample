import os
from time import sleep

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
    if DEBUG == "1":
        print(
            "Adding metadata for {} from UDF to host: {} and port: {}".format(
                options["Name"], options["host"], options["port"]
            ),
            flush=True,
        )
    db = vdms.vdms()
    db.connect(options["host"], options["port"])

    if DEBUG == "1":
        print(
            "[DEBUG add_metadata] {} metadata keys: {}".format(
                options["Name"], list(options["metadata"].keys())
            ),
            flush=True,
        )

    ref = 1
    query = [
        {
            "FindVideo": {
                "_ref": ref,
                "constraints": {
                    # "uid": ["==", options["uid"]],
                    "Name": ["==", options["Name"]],
                },
                "results": {"limit": 1, "list": ["uid", "Name"]},
            }
        }
    ]
    # response = retry_query(db, query, num_retries=10, sleep_timer=5)
    # query[0]["FindVideo"]["_ref"] = ref
    # query = []

    fref = 2
    # last_frame = -1
    add_query_ref = 0
    for k in options["metadata"].keys():
        metadata = options["metadata"][k]
        # fref += 2
        # curr_frame = metadata["frame_props"]["frameID"]
        # print(f"[DEBUG add_metadata] curr_frame: {curr_frame} last_frame: {last_frame}", flush=True)
        # if last_frame != curr_frame:
        add_query = {
            "AddEntity": {
                "_ref": fref,
                "class": "Frame",
                "properties": metadata["frame_props"],
                "constraints": {
                    "server_filepath": ["==", options["Name"]],
                    "frameID": ["==", metadata["frame_props"]["frameID"]],
                },
                "link": {
                    "ref": 1,
                    "class": "Vid2Frame",
                    "properties": metadata["edge_props"],
                },
            }
        }
        query.append(add_query)
        add_query_ref = fref
        # last_frame = metadata["frame_props"]["frameID"]
        # vid_name = options["Name"]
        fref += 1
        # print(
        #     f"[DEBUG add_metadata] Frame {last_frame} added to {vid_name} query",
        #     flush=True,
        # )

        # add_frame_conn_query = {
        #     "AddConnection": {
        #         "class": "Vid2Frame",
        #         "properties": metadata["edge_props"],
        #         "ref1": 1,
        #         "ref2": fref,
        #     }
        # }
        # add_query = {
        #     "FindEntity": {
        #         "_ref": fref,
        #         "class": "Frame",
        #         "constraints": {"server_filepath": ["==", options["Name"]], "frameID": ["==",  metadata["frame_props"]["frameID"]]},
        #         # "link": {"ref": 1, "class": "Vid2Frame", "properties": metadata["edge_props"]},
        #     }
        # }

        add_bbox_query = {
            "AddBoundingBox": {
                "_ref": fref,  # fref + 1,
                "properties": metadata["bbox_props"],
                "rectangle": {
                    "h": int(metadata["bbox_props"]["VD:height"]),
                    "w": int(metadata["bbox_props"]["VD:width"]),
                    "x": int(metadata["bbox_props"]["VD:x1"]),
                    "y": int(metadata["bbox_props"]["VD:y1"]),
                },
                "link": {
                    "ref": add_query_ref,
                    "class": "Frame2BB",
                    "properties": metadata["bb_edge_props"],
                },
            }
        }

        # add_bbox_conn_query = {
        #     "AddConnection": {
        #         "class": "Frame2BB",
        #         "properties": metadata["bb_edge_props"],
        #         "ref1": fref,
        #         "ref2": fref + 1,
        #     }
        # }

        # query.append(add_query)
        # query.append(add_frame_conn_query)
        query.append(add_bbox_query)
        fref += 1
        # query.append(add_bbox_conn_query)

    # response, res_arr = db.query(query, [[]])
    response = retry_query(db, query, num_retries=10, sleep_timer=5)

    if DEBUG == "1":
        print(
            "[DEBUG] {} BACKGROUND ADD_METADATA RESPONSE: {}".format(
                options["Name"], response
            ),
            flush=True,
        )

    # db.disconnect()
    # del db
    return ipfilename, None
