import os

import vdms

DEBUG = os.environ.get("DEBUG", "0")
LOCKTIMEOUT_RETRIES = 5


def run(ipfilename, format, options, tmp_dir_path):
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
            "{} metadata keys: {}".format(
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
    for _ in range(LOCKTIMEOUT_RETRIES):
        response, _ = db.query(query, [[]])
        if "FailedCommand" in response[0] and "timeout" in response[0]["info"].lower():
            pass  # Rerun
        else:
            break  # Continue

    if DEBUG == "1":
        print(
            "[DEBUG] {} BACKGROUND ADD_METADATA RESPONSE: {}".format(
                options["Name"], response
            ),
            flush=True,
        )
    return ipfilename, None
