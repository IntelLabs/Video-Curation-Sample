# importing required libraries
import os
import subprocess
import sys
import time  # time library

# # REPO_DIR = Path(__file__).parent.parent
# # TEST_VIDEO_PATH = REPO_DIR / "video/archive_custom/video8K__test-8k-26s.mp4"
# SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
# # Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
tmp_dir = "/var/www/archive"
# kkhost = os.environ["KKHOST"]
# dbhost = "vdms-service"  # os.environ["DBHOST"]
# dbport = 55555
# ingestion = os.environ["INGESTION"]
in_source = os.environ["IN_SOURCE"]
# resize_input = str2bool(os.getenv("RESIZE_FLAG", False))
DEBUG = os.environ["DEBUG"]
DEBUG_FLAG = True if DEBUG == "1" else False
# # video_store_dir = "/home/resources"
# video_store_dir = "/var/www/mp4"
# model_w, model_h = (640, 640)
# DEVICE = os.environ["DEVICE"]


def main(watch_folder=os.getcwd()):
    if DEBUG_FLAG:
        print("[TIMING],start_watchandsend,," + str(time.time()), flush=True)

    print(f"in_source: {in_source}", flush=True)
    if "videos" in in_source:
        # sorted_files = sort_files_in_directory_by_size("/var/www/archive")
        # all_processes = []
        for filename in os.listdir(tmp_dir):
            if any(filename.endswith(ext) for ext in [".mp4", ".mkv", ".avi"]):
                full_filename_path = os.path.join(tmp_dir, filename)
                # file_prefix = Path(full_filename_path).stem
                # processor(full_filename_path)
                print(f"Starting processing for {full_filename_path}", flush=True)
                cmd = [sys.executable, "/home/process_stream.py", full_filename_path]
                process = subprocess.Popen(
                    cmd,
                    #    stdout=subprocess.PIPE,
                    #    stderr=subprocess.STDOUT
                )
                process.wait()
                # all_processes.append(process)
                # try:
                #     result = subprocess.run(cmd, capture_output=True, text=True)
                #     print("Script output:", result.stdout)
                #     print("Script errors:", result.stderr)
                #     # result = subprocess.run(
                #     #     cmd, # Use sys.executable to ensure the correct Python interpreter
                #     #     check=True, # Raise a CalledProcessError if the script returns a non-zero exit code
                #     #     capture_output=True,
                #     # )
                #     # print(result)
                # except Exception:
                #     e = traceback.format_exc()
                #     print(f"Error occurred: {e}")

                # except subprocess.CalledProcessError as e:
                #     # e = traceback.format_exc()
                #     output = e.output
                #     print(f"Error occurred: {output}")

        # for p in all_processes:
        #     p.wait()

    if "stream" in in_source:
        import yaml

        with open("/home/camera_config.yaml", "r") as inFile:
            config = yaml.safe_load(inFile)
        for camera_name, camera_details in config.items():
            print(f"Starting processing for {camera_name}", flush=True)
            cmd = [
                sys.executable,
                "/home/process_stream.py",
                camera_details["url"],
                camera_name,
            ]
            process = subprocess.Popen(
                cmd,
            )
            process.wait()
    if DEBUG_FLAG:
        print("[TIMING],end_watchandsend,," + str(time.time()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError("Invalid input. Please provide video path or camera URL")
