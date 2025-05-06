import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from zipfile import ZipFile

import cv2
import numpy as np
from flask import Flask, after_this_request, jsonify, request, send_file
from werkzeug.utils import secure_filename

tmp_dir_path = None


# Function to dynamically import a module given its full path
def import_module_from_path(module_name, path):
    try:
        # Create a module spec from the given path
        spec = importlib.util.spec_from_file_location(module_name, path)

        # Load the module from the created spec
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print("import_module_from_path() failed:", str(e))
        return None


def setup(tmp_path):
    global tmp_dir_path

    # Get the real directory where this Python file is
    currentDir = os.path.realpath(os.path.dirname(__file__))

    if tmp_path is None:
        tmp_path = currentDir
        print("Warning: Using temporary dir:", tmp_path, " as default.")

    if not os.path.exists(tmp_path):
        raise Exception(f"{tmp_path}: path to temporary dir is invalid")

    functions_path = os.path.join(currentDir, "functions")
    if not os.path.exists(functions_path):
        raise Exception(f"{functions_path}: path to functions dir is invalid")

    # Set path to temporary dir
    tmp_dir_path = tmp_path

    for entry in os.scandir(functions_path):
        if entry.is_file() and entry.path.endswith(".py"):
            module_name = entry.name[:-3]

            # Import the module from the given path
            module = import_module_from_path(module_name, entry)
            if module is None:
                raise Exception(
                    "setup() error: module '" + str(entry) + "' could not be loaded"
                )
            globals()[module_name] = module


app = Flask(__name__)


def get_current_timestamp():
    dt = datetime.now(timezone.utc)

    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()

    return utc_timestamp


@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"response": "true"})


@app.route("/image", methods=["POST"])
def image_api():
    global tmp_dir_path
    try:
        json_data = json.loads(request.form["jsonData"])
        image_data = request.files["imageData"]

        format = json_data["format"] if "format" in json_data else "jpg"

        tmpfile = secure_filename(
            os.path.join(tmp_dir_path, "tmpfile" + uuid.uuid1().hex + "." + str(format))
        )

        image_data.save(tmpfile)

        r_img, r_meta = "", ""

        if "id" not in json_data:
            raise Exception("id value was not found in json_data")

        id = json_data["id"]

        if id not in globals():
            raise Exception(f"id={id} value was not found in globals()")

        udf = globals()[id]

        if "ingestion" in json_data:
            r_img, r_meta = udf.run(tmpfile, format, json_data, tmp_dir_path)
        else:
            r_img, _ = udf.run(tmpfile, format, json_data, tmp_dir_path)

        img_encode = cv2.imencode("." + str(format), r_img)[1]

        # Converting the image into numpy array
        data_encode = np.array(img_encode)

        # Converting the array to bytes.
        return_string = data_encode.tobytes()

        if r_meta != "":
            return_string += ":metadata:".encode("utf-8")
            return_string += r_meta.encode("utf-8")

        os.remove(tmpfile)

        if return_string == "" or return_string is None:
            return "error"

        return return_string
    except Exception as e:
        error_message = f"Exception: {str(e)}"
        print(error_message, file=sys.stderr)
        return "An internal error has occurred. Please try again later."


@app.route("/video", methods=["POST"])
def video_api():
    global tmp_dir_path
    try:
        json_data = json.loads(request.form["jsonData"])
        video_data = request.files["videoData"]
        format = json_data["format"] if "format" in json_data else "mp4"
        input_sizeWH = json_data["input_sizeWH"]

        tmpfile = secure_filename(
            os.path.join(tmp_dir_path, "tmpfile" + uuid.uuid1().hex + "." + str(format))
        )
        video_data.save(tmpfile)

        video_file, metadata_file = "", ""

        if "id" not in json_data:
            raise Exception("id value was not found in json_data")

        id = json_data["id"]

        if id not in globals():
            raise Exception(f"id={id} value was not found in globals()")

        udf = globals()[id]

        if "ingestion" in json_data:
            video_file, metadata_file = udf.run(
                tmpfile, format, json_data, tmp_dir_path, input_sizeWH
            )
        else:
            video_file, _ = udf.run(tmpfile, format, json_data, tmp_dir_path, input_sizeWH)

        response_file = os.path.join(
            tmp_dir_path, "tmpfile" + uuid.uuid1().hex + ".zip"
        )

        with ZipFile(response_file, "w") as zip_object:
            zip_object.write(video_file)
            if metadata_file != "":
                zip_object.write(metadata_file)

        os.remove(tmpfile)

        # Delete the temporary files after the response is sent
        @after_this_request
        def remove_tempfile(response):
            try:
                os.remove(response_file)
                os.remove(video_file)
                os.remove(metadata_file)
            except Exception:
                print("Warning: Some files cannot be deleted or are not present")
            return response

        try:
            return send_file(
                response_file, as_attachment=True, download_name=response_file
            )
        except Exception as e:
            print("Error in file read:", str(e), file=sys.stderr)
            return "Error in file read"
    except Exception:
        return "An internal error has occurred. Please try again later."


@app.errorhandler(400)
def handle_bad_request(e):
    response = e.get_response()
    response.data = json.dumps(
        {
            "code": e.code,
            "name": e.name,
            "description": e.description,
        }
    )
    response.content_type = "application/json"
    return response


def main():
    num_args = len(sys.argv)
    if sys.argv[1] is None:
        print("Port missing\n Correct Usage: python3 udf_server.py <port> [tmp_path]")
    elif num_args > 2 and sys.argv[2] is None:
        print(
            "Warning: Path to the temporary directory is missing\nBy default the path will be the current directory"
        )
        setup(None)
        app.run(host="0.0.0.0", port=int(sys.argv[1]))
    else:
        setup(sys.argv[2])
        app.run(host="0.0.0.0", port=int(sys.argv[1]))


if __name__ == "__main__":
    main()
