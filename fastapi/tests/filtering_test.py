import argparse
import os
import queue
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2

sys.path.insert(1, str(Path(__file__).parent.parent))
# os.environ["ENABLE_VDMS"] = False


from include.handlers import (
    CLIP_DURATION,
    RAW_BB_FULL_RES_PADDING,
    HybridReader,
)
from include.handlers import VideoStreamHandler_WIP as VideoStreamHandler

# merge_boxes_limit, MERGE_SIZE_LIMIT
# MERGE_SIZE_LIMIT = MODEL_W # MODEL_W, 960
ENABLE_QUERYING = False
TARGET_FPS = os.getenv("TARGET_FPS", 15)
THICKNESS = 3


# def fast_filter_boxes(boxes, thresh=0.5):
#     if len(boxes) == 0:
#         return []

#     # Convert to np.array [x1, y1, x2, y2]
#     boxes = np.array(boxes)

#     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     areas = (x2 - x1) * (y2 - y1)

#     # Sort by area (Largest first)
#     order = areas.argsort()[::-1]
#     keep = []

#     while order.size > 0:
#         i = order[0]
#         keep.append(boxes[i])

#         # Find intersection with all other boxes
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h

#         # Calculate ratio of intersection to the SMALLER boxes' areas
#         # This identifies containment rather than just overlap (IoU)
#         overlap = inter / areas[order[1:]]

#         # Keep indices where overlap is less than threshold
#         inds = np.where(overlap < thresh)[0]
#         order = order[inds + 1]

#     return keep


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v",
        "--test-video",
        type=str,
        dest="source",
        required=True,
        help="Test video path for prediction.",
    )

    parser.add_argument(
        "-r",
        "--range",
        type=int,
        nargs=2,
        dest="debug_range",
        default=[6, 7],
        help="Range of frames to save mask images ([min,max]).",
    )

    args = parser.parse_args()
    video_path = Path(__file__).parent / f"test_videos/{args.source}"
    args.source = video_path
    vid_name = video_path.stem
    args.out_imgdir = Path(__file__).parent / f"test_videos/{vid_name}_test_imgs"
    args.out_imgdir.mkdir(parents=True, exist_ok=True)
    return args


class TestHybridReader(HybridReader):
    def __init__(self, source, target_fps=TARGET_FPS, clip_duration=CLIP_DURATION):
        super().__init__(source, target_fps=target_fps, clip_duration=clip_duration)

        self.frame_idx = 0
        self.target_frame_idx = 0
        self.frame_queue = queue.Queue(maxsize=30)

    def update(self):
        while not self.stopped:
            # Determine if current frame_idx should be "KEPT" or "SKIPPED"
            # to match the target cadence
            should_keep = int(self.frame_idx * self.target_fps / self.input_fps) > int(
                (self.frame_idx - 1) * self.target_fps / self.input_fps
            )

            if should_keep:
                # Fully decode this frame
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
                self.target_frame_idx += 1
                print(f"Target Frame {self.target_frame_idx} in queue\n", flush=True)
            else:
                # Fast-forward the pointer without decoding (minimal CPU)
                self.cap.grab()

            self.frame_idx += 1

    def read(self):
        try:
            # If the reader is stopped, don't wait a full second;
            # check immediately to speed up the "Draining" phase.
            wait_time = 0.1 if self.stopped else 1.0
            return self.frame_queue.get(timeout=wait_time)
        except queue.Empty:
            return None


class TestVideoStreamHandler(VideoStreamHandler):
    # Override VideoStreamHandler.__init__ for debugging
    def __init__(self, source, name, active_streams, **kwargs):
        """
        Initializes the pipeline and pre-allocates isolated memory buffers.

        Args:
            source (str): The RTSP URL or local file path.
            name (str): Unique identifier for the stream.
            active_streams (dict): Global dictionary tracking all running handlers.
        """
        # Initialize BaseHandler
        super().__init__(source, name, active_streams, **kwargs)

        self.debug_range = kwargs.get("debug_range", [6, 7])

    # Override VideoStreamHandler.setup_threads to avoid threads for debugging
    def setup_threads(self):
        pass

    # Override VideoStreamHandler.start to avoid threads for debugging
    def start(self):
        # Start the hardware-decoupled reader first
        self.reader.start()

        # self.process_thread.start()
        self.run_filtering()

    # Override VideoStreamHandler.stop
    def stop(self):
        """
        Comprehensive resource release.
        """
        with self._stop_lock:
            if self._is_stopped:
                return  # Already stopped by another thread

        # Signal threads to stop
        self.active = False

        # Close the OpenCV capture
        if self.cap:
            self.cap.release()
            self.cap = None

        # Stop reader thread
        if hasattr(self, "reader"):
            self.reader.stop()

        # This ensures the next /dashboard_stats call won't see this stream
        if self.name in self.active_streams:
            self.active_streams.pop(self.name, None)

        # Final Reset of the FastAPI event
        self.frame_ready_event.set()  # Unblock any generators waiting on this stream

        # Purge HW Buffers
        self._is_stopped = True
        if self.device_input == "cuda":
            self.cleanup_gpu()
        else:
            self.cleanup_cpu()

    # ----- FRAME PROCESSING -----
    # Override VideoStreamHandler.process_frame_async for testing and retrieving/writing annotated frame
    def process_frame_async(self, frame, frame_num):
        """
        Worker function to run heavy AI tasks (Resize, Bkgd Sub, YOLO)
        in the background without blocking the video reader.
        """
        # print(f"Processing frame {frame_num}", flush=True)
        annotated_frame = None
        inf_data = None
        try:
            if self.device_input == "cuda":
                inf_data = self.test_rbtd_detection_gpu(frame, frame_num)
            else:
                inf_data = self.test_full_cpu(frame, frame_num)

            if inf_data:
                mask_2_write = (
                    inf_data["mask"].download(self.stream)
                    if self.device_input != "cpu"
                    else inf_data["mask"]
                )
                frame_2_write = (
                    self.resized_frame.download(self.stream)
                    if self.device_input != "cpu"
                    else self.cpu_resized_frame
                )
                if frame_num > (self.debug_range[0] * self.target_fps) and frame_num < (
                    self.debug_range[1] * self.target_fps
                ):
                    cv2.imwrite(
                        f"{self.out_imgdir}/frameNum_{frame_num}_resized_mask.png",
                        mask_2_write,
                    )
                    cv2.imwrite(
                        f"{self.out_imgdir}/frameNum_{frame_num}_resized_orig.png",
                        frame_2_write,
                    )

                annotated_frame = self.contour2annotatedframe(
                    inf_data["frameNum"],
                    mask_2_write,
                    inf_data["full_frame"],
                    device_input=self.device_input,
                )

            if annotated_frame is not None:
                self.video_writer.write(annotated_frame)
                if frame_num > (self.debug_range[0] * self.target_fps) and frame_num < (
                    self.debug_range[1] * self.target_fps
                ):
                    cv2.imwrite(
                        f"{self.out_imgdir}/frameNum_{frame_num}_annotated_orig.png",
                        annotated_frame,
                    )
            else:
                self.video_writer.write(frame)

        except Exception:
            e = traceback.format_exc()
            print(
                f"ERROR: process_frame_async failed for {self.name}: {e}\n", flush=True
            )

    # Replaces VideoStreamHandler.run_realtime_inference specifically for filtering test
    def run_filtering(self):
        """
        Main loop: Initializes the model in this thread to fix CUDA context issues.
        """
        print(f"Inference thread started for {self.name}...\n", flush=True)

        while self.active:
            frame = self.reader.read()

            # Get current depth to monitor the backlog
            q_depth = self.reader.frame_queue.qsize()

            if frame is not None:
                self.frame_count += 1
                print(
                    f"Processing frame {self.frame_count} | Queue Depth: {q_depth}\n",
                    flush=True,
                )
                self.process_frame_async(frame, self.frame_count)

                self.last_heartbeat = time.time()

            else:
                # Check if we are actually done or just waiting for the reader
                if self.reader.stopped:
                    if q_depth > 0:
                        print(
                            f" [WAIT] Reader stopped, but {q_depth} frames remain. Draining...\n",
                            flush=True,
                        )
                        continue  # Keep looping to pull remaining frames
                    else:
                        print(
                            f" [FINISHED] Total frames processed: {self.frame_count}\n",
                            flush=True,
                        )
                        break
                else:
                    # The queue is temporarily empty because the reader is slow
                    time.sleep(0.001)

        self.video_writer.release()
        self.encode_video()
        self.stop()

    # ----- ROI RELATED -----
    # Used to retrieve annotated full resolution frame from contours
    # Called from within contour2annotatedframe
    # Comparable to VideoStreamHandler.get_detections_for_contours_bbs
    def get_annotation_for_contours_bbs(
        self,
        frameNum,
        foi,
        contours,
        color_mask=None,
        thickness=THICKNESS,
        device_input="cuda",
    ):
        H, W = foi.shape[:2]  # Unpack once

        if not contours:
            return foi

        # Filter small noise and convert contours to 8K-space bounding boxes
        raw_bbs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.min_contour_area:
                x1, y1, w, h = cv2.boundingRect(c)

                if (
                    color_mask is not None
                    and frameNum > (self.debug_range[0] * self.target_fps)
                    and frameNum < (self.debug_range[1] * self.target_fps)
                ):
                    # color_mask = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
                    color_mask = cv2.rectangle(
                        color_mask, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), thickness
                    )  # BGR- RED

                # Scale coordinates from 640p BGS-space to 8K-space
                xx1 = max(0, int((x1 * self.scale_x)) - RAW_BB_FULL_RES_PADDING)
                yy1 = max(0, int((y1 * self.scale_y)) - RAW_BB_FULL_RES_PADDING)
                xx2 = min(W, int(((x1 + w) * self.scale_x)) + RAW_BB_FULL_RES_PADDING)
                yy2 = min(H, int(((y1 + h) * self.scale_y)) + RAW_BB_FULL_RES_PADDING)
                raw_bbs.append([area, [xx1, yy1, xx2, yy2]])

        dist_thresh = min(0.05 * W, 0.05 * H)
        merged = self.filter_rois(raw_bbs, dist_thresh=dist_thresh)

        # Extract crops at full-resolution
        crop_cnt = 0
        for x1, y1, x2, y2 in merged:
            if (
                (x2 - x1) > 31
                and (y2 - y1) > 31
                and (x2 - x1) < self.frame_width
                and (y2 - y1) < self.frame_height
            ):
                crop_cnt += 1
                foi = cv2.rectangle(foi, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        if (
            color_mask is not None
            and frameNum > (self.debug_range[0] * self.target_fps)
            and frameNum < (self.debug_range[1] * self.target_fps)
        ):
            cv2.imwrite(
                f"{self.out_imgdir}/frameNum_{frameNum}_annotated_resized_mask.png",
                color_mask,
            )

        return foi

    # Used to retrieve annotated full resolution frame from motion mask
    # Comparable to VideoStreamHandler.contour2predictions
    def contour2annotatedframe(self, frameNum, mask, frame, device_input="cpu"):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, mask = self.get_reduced_contour(mask, contours)

        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        annotated_frame = self.get_annotation_for_contours_bbs(
            frameNum,
            frame,
            contours,
            color_mask=color_mask,
            thickness=THICKNESS,
            device_input=device_input,
        )

        return annotated_frame

    # ----- Result Video Re-encoding -----
    # Re-encode video with ffmpeg
    def encode_video(self):
        print("Encoding video ...\n", flush=True)
        # Re-encode video in order to seek via ffmpeg later
        GENERAL_OPTS = "-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"  #  -filter:v fps={target_fps}
        CONVERSION = f"-c:v libx264 -preset ultrafast -filter:v fps=fps={self.target_fps}"  # "-c:v libx264 -preset medium"
        reencode_cmd = f"ffmpeg -y -i {self.tmp_file} {GENERAL_OPTS} {CONVERSION} -crf 23 -c:a copy {self.clip_filename}"
        try:
            cmd_list = shlex.split(reencode_cmd)

            subprocess.run(cmd_list, check=True)

            # Cleanup the temporary RAM-disk file immediately
            if os.path.exists(self.tmp_file):
                os.remove(self.tmp_file)

        except Exception as e:
            print(f" [ERROR] Clip finalization failed: {e}\n", flush=True)


def main(args):
    active_stream = TestVideoStreamHandler(
        args.source,
        args.source.stem,
        {},
        target_fps=TARGET_FPS,
        debug_range=args.debug_range,
    )

    active_stream.out_imgdir = str(args.out_imgdir)
    active_stream.clip_filename = str(args.source).replace(".mp4", "_annotated.mp4")
    tmp_file = Path(active_stream.clip_filename).name
    active_stream.tmp_file = f"/dev/shm/{tmp_file}"

    active_stream.video_writer = cv2.VideoWriter(
        active_stream.tmp_file,
        active_stream.fourcc,
        active_stream.target_fps,
        (active_stream.frame_width, active_stream.frame_height),
    )

    active_stream.start()


if __name__ == "__main__":
    in_params = get_input_args()
    main(in_params)
