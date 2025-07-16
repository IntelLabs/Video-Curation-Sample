import json
import os
import time
import uuid
import vdms

def run(ipfilename, format, options, tmp_dir_path):
    print("Adding metadata from UDF to host: {} and port: {}".format(options['host'], options['port']))
    db = vdms.vdms()
    db.connect(options['host'], options['port'])

    print(options["metadata"].keys())

    ref = 1
    query = [
        {
            "FindVideo": {
                "_ref": ref,
                "constraints" : {
                    "uid" : ["==", options['uid']],
                },
                "results":{
                    "limit":1
                }
            }
        }
    ]
    fref = 0
    for k in options["metadata"]:
        metadata = options["metadata"][k]
        fref+=2
        add_query = {
            "AddEntity" : {
                "_ref": fref,
                "class": "Frame",
                "properties": metadata["frame_props"]
            }
        }
        
        add_frame_conn_query = {
            "AddConnection": {
                "class": "Vid2Frame",
                "properties": metadata["edge_props"],
                "ref1": 1,
                "ref2": fref
            }
        }

        add_bbox_query = {
            "AddBoundingBox" : {
                "_ref": fref+1,
                "properties": metadata["bbox_props"],
                "rectangle": {
                    "h": int(metadata["bbox_props"]["VD:height"]),
                    "w": int(metadata["bbox_props"]["VD:width"]),
                    "x": int(metadata["bbox_props"]["VD:x1"]),
                    "y": int(metadata["bbox_props"]["VD:y1"]),
                }
            }
        }

        add_bbox_conn_query = {
            "AddConnection": {
                "class": "Frame2BB",
                "properties": metadata["bb_edge_props"],
                "ref1": fref,
                "ref2": fref+1
            }
        }

        query.append(add_query)
        query.append(add_frame_conn_query)
        query.append(add_bbox_query)
        query.append(add_bbox_conn_query)

    response, res_arr = db.query(query, [[]])
    print(response)
    return ipfilename, None