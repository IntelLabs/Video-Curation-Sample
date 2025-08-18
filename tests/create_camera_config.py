# THIS FILE CREATES MOCK CAMERA_CONFIG.YAML FILES FOR TESTING

import argparse
from pathlib import Path

import yaml

PROJECT_PATH = Path(__file__).parent.parent


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        default=PROJECT_PATH / "tests/results/camera_configs",
        type=Path,
        help="Path for result directory",
    )

    parser.add_argument(
        "-n",
        "--num-cameras",
        dest="camera_counts",
        # type=int,
        nargs="+",
        help="Number of cameras to include in config",
    )

    parser.add_argument(
        # "-h",
        "--host",
        type=str,
        help="RTSP host for cameras",
    )

    parser.add_argument(
        # "-p",
        "--port",
        type=int,
        default=8554,
        help="RTSP port for cameras",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    return args


def main(args):
    for camera_count in args.camera_counts:
        # Create dict for yaml file
        camera_details = {}
        for cam_num in range(1, int(camera_count) + 1):
            camera_name = f"rtsp{cam_num}"
            camera_details[camera_name] = {
                "type": "rtsp",
                "url": f"rtsp://{args.host}:{args.port}/{camera_name}",
            }

        # Write to YAML
        file_path = args.out_dir / f"camera_config_{camera_count}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(camera_details, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    args = get_input_args()
    main(args)
