# ==============================================================================
# LOGGING

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

main_app_logger = logging.getLogger(__name__)


# ==============================================================================
# IMPORTS

import time
import traceback
from collections import deque
from pathlib import Path

import cupy
import cupyx.scipy
import cupyx.scipy.ndimage
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.engine.results import Boxes, Results
from ultralytics.utils.nms import non_max_suppression

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.models import get_model
from include.utils import (
    get_bb_overlay,
    get_bounds_kernel,
    merge_boxes_gpu,
)

# ==============================================================================
# CONFIGURATIONS
STREAM_ARG = False
PADDING_SCALE = 0.5


class BaseObjectDetector:
    def __init__(
        self,
        config,
        device="cuda",
        timer_enabled=True,
        resize_hw=(640, 640),
        frame_hw=(4320, 7680),
        target_fps=15.0,
        result_dir="/tmp",
        run_name=None,
        debug_frame_limit=-1,
        **kwargs,
    ):  # , label_source=["drone"]
        self.device_input = device
        self.timer_enabled = timer_enabled
        self.config = config
        self.target_fps = target_fps
        self.resize_h, self.resize_w = int(resize_hw[0]), int(resize_hw[1])
        self.frame_height, self.frame_width = int(frame_hw[0]), int(frame_hw[1])
        # self.label_source = label_source
        self.compiled_no_grad_gate = torch.no_grad()

        if run_name is None:
            run_type = "sf" if self.config.sf_enabled else "yolo"
            det_type = self.config.DETECTION_TYPE
            device_name = self.config.DEVICE.lower()
            run_name = f"{run_type}_{det_type}_{device_name}"
        self._testMethodName = run_name

        self.result_dir = result_dir
        # self.debug_frame_limit = debug_frame_limit
        self.debug_frame_limit = (
            debug_frame_limit
            if self.config.DEBUG_FLAG and debug_frame_limit > -1
            else -1
        )

        provided_model = kwargs.get("model")
        self.setup_model(provided_model)
        self.initialize()

    def model_warmup(self, H=640, W=640):
        H, W = int(H), int(W)
        # Move the dummy input creation inside a no_grad block
        with self.compiled_no_grad_gate:
            # main_app_logger.info(f"Starting warmup for {self.name}...")
            dummy_input = torch.zeros((1, 3, H, W)).to(self.device_input)

            # Perform iterations directly on the main thread
            for i in range(5):
                dummy_result = self.run_model(
                    dummy_input,
                    imgsz=(H, W),
                    batch=1,
                    device_input=self.device_input,
                    stream=STREAM_ARG,
                )
                # if i == 0 and hasattr(self.model, "predictor") and self.model.predictor:
                #     self.cached_predictor = self.model.predictor
                #     self.cached_predictor.profile = False

            del dummy_input, dummy_result

            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                predictor = self.model.predictor

                # Disable the internal framework statistics stopwatch loops
                # This wipes out the trailing 2.3ms context synchronization gates completely!
                predictor.stride = 32
                predictor.profile = False

                # CRITICAL OPTIMIZATION: Tell the post-processor to skip input image array compilation.
                # This completely satisfies the internal checks, bypassing convert_torch2numpy_batch
                # and dropping your 17.99ms PCIe bus device-to-host download stall down to 0 ms.
                predictor.save_dir = None
                predictor.args.visualize = False
                predictor.args.show = False
                predictor.args.save = False
                self.cached_predictor = predictor

            # Force GPU to finish before returning
            # if self.device_input == "cuda":
            #     torch.cuda.synchronize()

            # Pin a lightweight tensor to prevent downstream empty_cache() calls
            # from destroying the compiled model weight layouts in memory
            self.__persistent_vram_lock = torch.zeros((1,), device=self.device_input)

        main_app_logger.info("Warmup complete")

    def setup_model(self, provided_model, force_export=False):
        if (
            self.frame_width * self.frame_height
        ) <= self.config.SMART_FILTERING_PIXEL_CONSTRAINT:
            if "_noSF" not in self.config.model_path:
                oldpath = Path(self.config.model_path)
                old_modelname = self.config.MODEL_NAME
                self.config.MODEL_NAME = f"{old_modelname}_noSF"
                new_model_name = oldpath.name.replace(
                    old_modelname, self.config.MODEL_NAME
                )
                self.config.model_path = str(oldpath.parent / new_model_name)

        if provided_model is not None and not isinstance(provided_model, str):
            self.model = provided_model
            self.label_source = [v for k, v in self.model.names.items()]
            self.label_source_dict = self.model.names
        else:
            run_platform_name = "engine" if "cuda" in self.device_input else "openvino"
            self.model, _, self.label_source = get_model(
                Path(self.config.model_path).parent,
                self.config.MODEL_NAME.replace("_noSF", ""),
                run_platform_name,
                self.device_input,
                batch=self.config.MODEL_MAX_BATCH_SIZE,
                force_export=force_export,
                sf_enabled=self.config.sf_enabled,
                model_h=self.resize_h,
                model_w=self.resize_w,
            )
            self.label_source_dict = self.model.names

            # if run_platform_name == "openvino":
            #     self.model.predictor.args.embed = False

            # Warmup Model
            if not self.config.sf_enabled:
                self.model_warmup(self.frame_height, self.frame_width)
            else:
                self.model_warmup(self.resize_h, self.resize_w)

    def run_model(
        self,
        frame,
        imgsz=(640, 640),
        batch=1,
        device_input="cuda",
        stream=False,
    ):
        if (
            isinstance(frame, torch.Tensor)
            # and device_input == "cuda"
            # and getattr(self, "cached_predictor", None) is not None
        ):
            if frame.dtype == torch.uint8:
                # Use standard out-of-place division to prevent byte truncation errors
                frame = frame.to(dtype=torch.float32).mul(1.0 / 255.0)
            else:
                # If it's already a float tensor array block, in-place scaling is safe
                frame = frame.mul_(1.0 / 255.0)
            raw_module = None

            if hasattr(self, "cached_predictor") and self.cached_predictor is not None:
                raw_module = self.cached_predictor.model

            with torch.inference_mode():
                # Establish your pipeline invariant:
                # frame = [B,C,H,W], uint8, CUDA, [0,255].
                # if frame.dtype == torch.uint8:
                #     dtype = (
                #         torch.float16
                #         if getattr(raw_module, "fp16", False)
                #         else torch.float32
                #     )
                #     frame = frame.to(dtype=dtype).mul_(1.0 / 255.0)
                # elif not frame.is_floating_point():
                #     frame = frame.float()

                if raw_module is not None:
                    # start = torch.cuda.Event(enable_timing=True)
                    # end = torch.cuda.Event(enable_timing=True)

                    # start.record()
                    # If input is guaranteed 640x640, don't pad here.
                    raw_output = raw_module(frame)
                    # end.record()
                    # torch.cuda.synchronize()
                    # inference_ms = start.elapsed_time(end)

                    # start.record()
                    decoded_results = non_max_suppression(
                        prediction=raw_output,
                        conf_thres=self.config.DETECTION_THRESHOLD,
                        iou_thres=self.config.IOU_THRESHOLD,
                        max_det=self.config.MAX_DETECTIONS,
                        classes=None,
                    )
                    # end.record()
                    # torch.cuda.synchronize()
                    # nms_ms = start.elapsed_time(end)

                    # main_app_logger.info(f"Inference: {inference_ms:.3f} ms")
                    # main_app_logger.info(f"NMS:       {nms_ms:.3f} ms")

                    results = []
                    dummy_img = np.empty(
                        (1, 1, 3),
                        dtype=np.uint8,
                    )

                    for pred in decoded_results:
                        if pred is None or pred.shape[0] == 0:
                            pred = torch.empty(
                                (0, 6),
                                dtype=frame.dtype,
                                device=frame.device,
                            )

                        pre_calculated_boxes = Boxes(
                            pred,
                            imgsz,
                        )

                        # Construct a conforming container using native VRAM references
                        res = Results(
                            orig_img=dummy_img,
                            path="",
                            names=self.label_source_dict,
                            boxes=None,  # Bypasses internal allocation thrashing!  # pred,
                        )
                        res.boxes = pre_calculated_boxes
                        results.append(res)

                    del decoded_results, raw_output
                    return results

        # Fallback to normal Ultralytics path.
        with torch.inference_mode():
            if getattr(self, "cached_predictor", None) is not None:
                return self.cached_predictor(frame)
            else:
                static_img_h = int(imgsz[0])  # Maps explicitly to your fixed 640 target
                static_img_w = int(imgsz[1])
                return self.model.predict(
                    frame,
                    imgsz=(static_img_h, static_img_w),
                    batch=batch,
                    device=device_input,
                    verbose=False,
                    stream=stream,
                    conf=self.config.DETECTION_THRESHOLD,
                    iou=self.config.IOU_THRESHOLD,
                    max_det=self.config.MAX_DETECTIONS,
                    rect=(batch == 1),
                    profile=False,
                )

    def format_bbs_and_frame_4_detection_v1(self, bbs_full_res, device_frame):
        clean_bbs = []

        if self.config.sf_enabled and bbs_full_res is not None:
            if torch.is_tensor(bbs_full_res):
                # Check hardware registry constraints dynamically
                if bbs_full_res.is_cuda:
                    # 🚀 GPU Target: Stage non-blocking fast copy into page-locked host memory
                    # via PCIe DMA background channels to completely bypass lock-stalls
                    # pinned_host_tensor = bbs_full_res.detach()#.to(
                    # #     device="cpu",
                    # #     non_blocking=True,
                    # #     memory_format=torch.contiguous_format,
                    # # )
                    # clean_bbs = pinned_host_tensor.numpy()

                    # Moving the data to the host via non_blocking=True allows the PCIe
                    # DMA engine to handle the transfer, freeing up your CPU thread context!
                    cpu_tensor = bbs_full_res.detach().to(
                        device="cpu", non_blocking=True
                    )

                    # Extract the numpy array safely without triggering a global stream sync fence
                    clean_bbs = cpu_tensor.numpy()
                else:
                    # 🚀 CPU Target: Direct view mapping without triggering device-transfer hooks
                    clean_bbs = bbs_full_res.detach().numpy()
            else:
                clean_bbs = np.array(bbs_full_res)

        # Layout tracking adjustments (Universal compatibility layer)
        det_frame = device_frame
        if torch.is_tensor(det_frame):
            det_frame = det_frame.contiguous()  # byte()

            if det_frame.ndim == 4:
                det_frame = det_frame.squeeze(0).permute(1, 2, 0)
            elif det_frame.shape == 3:
                det_frame = det_frame.permute(1, 2, 0)

        merged = clean_bbs if self.config.sf_enabled else None

        return merged, det_frame

    def format_bbs_and_frame_4_detection(self, bbs_full_res, device_frame):
        clean_bbs = bbs_full_res if self.config.sf_enabled else None

        # if self.config.sf_enabled and bbs_full_res is not None:
        #     if torch.is_tensor(bbs_full_res):
        #         # Check hardware registry constraints dynamically
        #         if bbs_full_res.is_cuda:
        #             # 🚀 GPU Target: Stage non-blocking fast copy into page-locked host memory
        #             # via PCIe DMA background channels to completely bypass lock-stalls
        #             # pinned_host_tensor = bbs_full_res.detach()#.to(
        #             # #     device="cpu",
        #             # #     non_blocking=True,
        #             # #     memory_format=torch.contiguous_format,
        #             # # )
        #             # clean_bbs = pinned_host_tensor.numpy()

        #             # Moving the data to the host via non_blocking=True allows the PCIe
        #             # DMA engine to handle the transfer, freeing up your CPU thread context!
        #             cpu_tensor = bbs_full_res.detach().to(device="cpu", non_blocking=True)

        #             # Extract the numpy array safely without triggering a global stream sync fence
        #             clean_bbs = cpu_tensor.numpy()
        #         else:
        #             # 🚀 CPU Target: Direct view mapping without triggering device-transfer hooks
        #             clean_bbs = bbs_full_res.detach().numpy()
        #     else:
        #         clean_bbs = np.array(bbs_full_res)

        # Layout tracking adjustments (Universal compatibility layer)
        det_frame = device_frame
        if torch.is_tensor(det_frame):
            # det_frame = det_frame.contiguous()  #byte()

            if det_frame.ndim == 4:
                det_frame = det_frame.squeeze(0)  # .permute(1, 2, 0)
            # elif det_frame.shape == 3:
            #     det_frame = det_frame.permute(1, 2, 0)

            if not det_frame.is_contiguous():
                det_frame = det_frame.contiguous()

        return clean_bbs, det_frame

    def motion2metadata(self, merged, frame_count_target):
        metadata = {}
        if merged is not None and merged.shape[0] > 0:
            # merged = merged / self.scales_tensor.cpu().numpy().reshape(-1, 4)
            merged = merged.div(self.scales_tensor.view(-1, 4))

            # Calculate area for each box: (xmax - xmin) * (ymax - ymin)
            widths = merged[:, 2] - merged[:, 0]
            heights = merged[:, 3] - merged[:, 1]
            areas = widths * heights
            # max_area = np.max(areas) if np.max(areas) > 0 else 1.0

            if torch.is_tensor(areas):
                # Use native PyTorch reduction math if areas is a device tensor block
                raw_max = torch.max(areas).item() if areas.numel() > 0 else 0.0
                max_area = raw_max if raw_max > 0 else 1.0
            else:
                # Fallback standard array parsing logic for host NumPy sequences
                raw_max = np.max(areas) if len(areas) > 0 else 0.0
                max_area = raw_max if raw_max > 0 else 1.0

            # Format motion results like detection results for evaluation
            for i, bb in enumerate(merged):
                disp_x, disp_y, disp_x2, disp_y2 = bb
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y

                if disp_w > 2 and disp_h > 2:
                    obj_id = len(metadata)
                    # num_objs += 1
                    framenum_str = f"{frame_count_target:04d}_{obj_id:04d}"
                    score = areas[i] / max_area

                    metadata[framenum_str] = {
                        "frameId": int(frame_count_target),
                        "bbId": framenum_str,
                        "bbox": {
                            "x": int(disp_x),
                            "y": int(disp_y),
                            "height": int(disp_h),
                            "width": int(disp_w),
                            "object": "",
                            "object_det": {
                                "confidence": score,
                                "frameH": int(self.resize_h),
                                "frameW": int(self.resize_w),
                            },
                        },
                    }

        return metadata

    def initialize(self):
        self.device = self.config.DEVICE
        self.device_input = self.config.device_input
        self.disp_w, self.disp_h = self.config.DISPLAY_FRAME_SIZE
        self.resize_h, self.resize_w = [self.config.MODEL_H, self.config.MODEL_W]
        # self.fixed_inference_batch = torch.empty(
        #     (
        #         self.config.MODEL_MAX_BATCH_SIZE,
        #         3,
        #         self.config.MODEL_H,
        #         self.config.MODEL_W,
        #     ),
        #     dtype=torch.half,
        #     device=self.device_input,
        # )

    # DEBUG FUNCTIONS --------------------------------------------
    def debug_save_mask(self, frame_source, frame_num, rois=None, gt_boxes=None):
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "mask"
        debug_dir.mkdir(parents=True, exist_ok=True)

        save_path = (
            debug_dir / f"frame_{frame_num:04d}_mask.jpg"
        )  # f"mask_{frame_num:04d}.jpg"

        # cv2.imwrite(
        #     # str(stage_debug_dir / f"frame_{f_num:04d}_stage6_threshold.jpg"),
        #     str(stage_debug_dir / f"frame_{self.frame_count_target:04d}_stagerois_merged.jpg"),
        #     display_frame,
        # )

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        # #  Download or copy the data
        # if hasattr(frame_source, "download"):
        #     img_cpu = frame_source.download()
        # elif torch.is_tensor(frame_source):
        #     # .contiguous() fixes the horizontal "shredding"/static look
        #     temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
        #     img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
        # else:
        #     # For numpy arrays (like your pinned memory), ensure memory is linear
        #     img_cpu = np.ascontiguousarray(frame_source)

        # #  Fix Visibility (Normalization)
        # # If float, scale to 0-255. If uint8, leave as is to avoid "neon" colors.
        # if img_cpu.dtype != np.uint8:
        #     if img_cpu.max() <= 1.0:
        #         img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
        #     else:
        #         img_cpu = img_cpu.astype(np.uint8)

        #  Download/Copy the frame
        if torch.is_tensor(frame_source):
            # .contiguous() is CRITICAL here to fix the "shredded" look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            # img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
            img_cpu = temp.contiguous().cpu().numpy()
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
        elif hasattr(frame_source, "download"):
            img_cpu = frame_source.download()
        else:
            img_cpu = np.ascontiguousarray(frame_source)

        #  Fix Shape: restore spatial grid if flattened
        if img_cpu.ndim == 3 and img_cpu.shape[0] == 1:
            img_cpu = img_cpu.reshape((self.resize_h, self.resize_w, 3))

        #  Fix Visibility: ONLY multiply if it's actually floating point
        # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
        if img_cpu.dtype != np.uint8:
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
            else:
                img_cpu = img_cpu.astype(np.uint8)

        # Color Space: Standardize to BGR for imwrite
        if len(img_cpu.shape) != 3:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

        h_img, w_img = img_cpu.shape[:2]
        scale_x = float(w_img) / self.frame_width
        scale_y = float(h_img) / self.frame_height
        if gt_boxes is not None:
            # Factor for converting 8K (original) bb dimensions to display dimensions
            # disp_w, disp_h = inf_data["mask"].shape[:2] if hasattr(inf_data["mask"], "shape") else inf_data["mask"].size()
            # scale_display_ox = disp_w / self.frame_width
            # scale_display_oy = disp_h / self.frame_height

            img_cpu = get_bb_overlay(
                img_cpu,
                gt_boxes,
                (scale_x, scale_y),
                (w_img, h_img),
                color=(0, 255, 0),  # green
            )

        # Draw 8K Boxes (Scaled down)
        if rois is not None and len(rois) > 0:
            boxes = rois.cpu().tolist() if torch.is_tensor(rois) else rois
            for box in boxes:
                x1, y1, x2, y2 = [
                    # int(box[0] * scale_x),
                    # int(box[1] * scale_y),
                    # int(box[2] * scale_x),
                    # int(box[3] * scale_y),
                    max(0, int(box[0] * scale_x)),
                    max(0, int(box[1] * scale_y)),
                    min(w_img - 1, int(box[2] * scale_x)),
                    min(h_img - 1, int(box[3] * scale_y)),
                ]
                cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save to disk
        cv2.imwrite(str(save_path), img_cpu)
        del img_cpu

    def debug_save_img_roi(self, frame_source, bbs_full_res, frame_num, gt_boxes=None):
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "img_roi"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # out_filename = str(debug_dir / f"analysis_{frame_num:04d}.jpg")
        save_path = debug_dir / f"frame_{frame_num:04d}_analysis.jpg"

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        #  Download/Copy the frame
        if torch.is_tensor(frame_source):
            # .contiguous() is CRITICAL here to fix the "shredded" look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
        elif hasattr(frame_source, "download"):
            img_cpu = frame_source.download()
        else:
            img_cpu = np.ascontiguousarray(frame_source)

        #  Fix Shape: restore spatial grid if flattened
        if img_cpu.ndim == 3 and img_cpu.shape[0] == 1:
            img_cpu = img_cpu.reshape((self.resize_h, self.resize_w, 3))

        #  Fix Visibility: ONLY multiply if it's actually floating point
        # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
        if img_cpu.dtype != np.uint8:
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
            else:
                img_cpu = img_cpu.astype(np.uint8)

        # Color Space: Standardize to BGR for imwrite
        if len(img_cpu.shape) != 3:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

        # Draw 8K Boxes (Scaled down)
        h_img, w_img = img_cpu.shape[:2]
        scale_x = w_img / self.frame_width
        scale_y = h_img / self.frame_height

        if gt_boxes is not None:
            # Factor for converting 8K (original) bb dimensions to display dimensions
            # disp_w, disp_h = inf_data["mask"].shape[:2] if hasattr(inf_data["mask"], "shape") else inf_data["mask"].size()
            # scale_display_ox = disp_w / self.frame_width
            # scale_display_oy = disp_h / self.frame_height

            img_cpu = get_bb_overlay(
                img_cpu,
                gt_boxes,
                (scale_x, scale_y),
                (w_img, h_img),
                color=(0, 255, 0),  # green
            )

        if bbs_full_res is not None:
            boxes = (
                bbs_full_res.cpu().tolist()
                if torch.is_tensor(bbs_full_res)
                else bbs_full_res
            )
            # Annotate Box to image shape
            for box in boxes:
                x1, y1, x2, y2 = [
                    max(0, int(box[0] * scale_x)),
                    max(0, int(box[1] * scale_y)),
                    min(w_img - 1, int(box[2] * scale_x)),
                    min(h_img - 1, int(box[3] * scale_y)),
                ]
                img_cpu = cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(save_path), img_cpu)
        del img_cpu

    def debug_save_crops(self, cropped_batch, frame_num):
        """Saves the first 5 crops of a batch to the results directory."""
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "crops"
        # debug_dir = self.result_dir / "debug_stages" / self._testMethodName
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Only save for the first self.config.DEBUG_FRAME_LIMIT frames to avoid disk bloat
        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        for i, crop in enumerate(cropped_batch[: self.config.DEBUG_FRAME_LIMIT]):
            # Convert GPU Tensor [C, H, W] -> NumPy [H, W, C]
            if torch.is_tensor(crop):
                # Reverse normalization (* 255) and permute to BGR
                # img = (crop.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
                temp = crop.squeeze(0) if crop.ndim == 4 else crop
                # Flip the color channels from RGB to BGR before any extraction utilities run
                # Flips only the first axis (channels) from [0, 1, 2] to [2, 1, 0]
                temp = torch.flip(temp, dims=[0])
                img = temp.permute(1, 2, 0).contiguous().cpu().numpy()
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = crop

            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                cv2.imwrite(str(debug_dir / f"frame_{frame_num}_crop_{i}.jpg"), img)

        del img

    def debug_save_img(self, frame_source, frame_num):
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "img"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        #  Download/Copy the frame
        if torch.is_tensor(frame_source):
            # .contiguous() is CRITICAL here to fix the "shredded" look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
        elif hasattr(frame_source, "download"):
            img_cpu = frame_source.download()
        else:
            img_cpu = np.ascontiguousarray(frame_source)

        #  Fix Shape: restore spatial grid if flattened
        if img_cpu.ndim == 3 and img_cpu.shape[0] == 1:
            img_cpu = img_cpu.reshape((self.resize_h, self.resize_w, 3))

        #  Fix Visibility: ONLY multiply if it's actually floating point
        # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
        if img_cpu.dtype != np.uint8:
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
            else:
                img_cpu = img_cpu.astype(np.uint8)

        # Color Space: Standardize to BGR for imwrite
        if len(img_cpu.shape) == 3:
            pass
        else:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(debug_dir / f"analysis_{frame_num:04d}.jpg"), img_cpu)
        del img_cpu


class GeneralObjectDetector(BaseObjectDetector):
    def initialize(self):
        super().initialize()

        if self.device_input == "cuda":
            self.gpu_id = 0
            self.device_index = f"cuda:{self.gpu_id}"
            self.inference_stream = torch.cuda.Stream()

            self.gpu_float_staging = torch.empty(
                (1, 3, self.frame_height, self.frame_width),
                dtype=torch.float16,
                device=self.device_index,
            )

            if self.timer_enabled:
                self.det_start, self.det_end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
        else:
            self.device_index = "cpu"

    def run(
        self, device_frame, overall_frame_num, frame_in_clip_count=0, gt_boxes=None
    ):
        if self.config.DETECTION_TYPE == "motion":
            main_app_logger.info(
                "[SKIP] Invalid type (motion) for GeneralObjectDetector!"
            )
            return

        metrics = {
            "sf_time": 0,
            "roi_time": 0,
            "det_time": 0,
            "bbs": None,
            "batch_density": 1,
        }
        inf_data = {}
        bbs_full_res = None
        motion_detected = True  # Uses all frames for evaluation
        inf_data["frameNum"] = overall_frame_num

        # Format BBs for Detection
        merged, det_frame = self.format_bbs_and_frame_4_detection(
            bbs_full_res, device_frame
        )

        try:
            if (
                self.config.DEBUG_FLAG
                and overall_frame_num <= self.config.DEBUG_FRAME_LIMIT
            ):
                self.debug_save_mask(
                    det_frame, overall_frame_num, rois=merged, gt_boxes=gt_boxes
                )

            if not self.config.DISABLE_DETECTION:
                # --- 3. MODEL INFERENCE TIMING BLOCK ---
                if self.device_input == "cuda" and self.timer_enabled:
                    self.det_start.record(self.inference_stream)
                elif self.timer_enabled:
                    t_start = time.perf_counter()

                metadata, _ = self.get_detections(
                    det_frame,
                    frame_in_clip_count,
                    device_input=self.config.device_input,
                )

                # num_objs = len(metadata.keys())

                if self.device_input == "cuda" and self.timer_enabled:
                    self.det_end.record(self.inference_stream)
                    # self.inference_stream.synchronize()

                    # Lock-free check: wait for event completion without blocking the CPU GIL
                    while not self.det_end.query():
                        # Yield GIL to let file-writers and thread-pools work
                        time.sleep(0.001)
                    # self.det_end.synchronize()
                    # Leverages the hardware driver scheduler without thread polling overhead
                    # if hasattr(self, "inference_stream"):
                    #     self.inference_stream.synchronize()

                    # Full-frame YOLO baseline tracks the elapsed time from t_start on page 20, line 1273
                    metrics["det_time"] = self.det_start.elapsed_time(self.det_end)

                elif self.timer_enabled:
                    # CPU Path Execution: Must use standard wall-clock timing loops to avoid CUDA Event errors
                    metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0

        except Exception:
            traceback.print_exc()
        finally:
            del merged, bbs_full_res
        return metrics, metadata, det_frame, motion_detected

    @torch.inference_mode()
    def get_detections(self, frame, frame_id, device_input="cuda"):
        metadata = {}
        num_objs = 0

        with torch.inference_mode():
            if isinstance(frame, torch.Tensor):
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    with self.compiled_no_grad_gate:
                        # 1. Transform from [4320, 7680, 3] to [3, 4320, 7680]
                        # 2. Unsqueeze(0) converts it to the required [1, 3, 4320, 7680] shape format
                        frame_inference = (
                            frame.permute(2, 0, 1).contiguous().unsqueeze(0)
                        )
                else:
                    frame_inference = frame.contiguous()
            else:
                frame_inference = frame

            H, W = frame_inference.shape[-2:]
            target_imgsz = (int(H), int(W))
            scale_display_x = self.resize_w / W  # 640 / 8192
            scale_display_y = self.resize_h / H  # 640 / 4608
            results = self.run_model(
                frame_inference,
                imgsz=target_imgsz,
                batch=1,
                device_input=device_input,
                stream=STREAM_ARG,
            )

        # Extract full resolution detections
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # OPTIMIZATION: Download coordinates as a unified batch array block to drop PCIe latency
                batch_coords = boxes.xywh.cpu().numpy()
                batch_cls = boxes.cls.cpu().numpy().astype(np.int32)
                batch_conf = boxes.conf.cpu().numpy()
                # for idx, box in enumerate(boxes):
                #     # coords = box.xywh[0].cpu().tolist()  # [x_center, y_center, width, height]
                #     # cls_id = int(box.cls[0].cpu().item())
                #     # conf = float(box.conf[0].cpu().item())
                #     coords = (
                #         box.xywh.cpu().squeeze().tolist()
                #     )  # Converts [x_center, y_center, w, h] safely
                #     cls_id = int(box.cls.cpu().item())
                #     class_name = self.label_source[cls_id]
                #     confidence = float(box.conf.cpu().item())
                for idx in range(len(batch_coords)):
                    coords = batch_coords[idx]
                    cls_id = batch_cls[idx]
                    class_name = self.label_source[cls_id]
                    confidence = float(batch_conf[idx])

                    # Guard against un-squeezed structural lists
                    if isinstance(coords[0], list):
                        coords = coords[0]

                    # Convert center bounds coordinates back to upper-left origin layout standard
                    # and scale to 640x640
                    disp_x = (coords[0] - (coords[2] / 2.0)) * scale_display_x
                    disp_y = (coords[1] - (coords[3] / 2.0)) * scale_display_y
                    disp_w = coords[2] * scale_display_x
                    disp_h = coords[3] * scale_display_y

                    # ----------------------------------
                    # TODO: Need function, same for SF and non-SF
                    if disp_w > 2 and disp_h > 2:
                        # Resized
                        object_res = [
                            int(disp_x),  # int(abs_x1 * scale_x),
                            int(disp_y),  # int(abs_y1 * scale_y),
                            int(disp_h),  # int(height * scale_y),
                            int(disp_w),  # int(width * scale_x),
                            class_name,
                            confidence,
                            int(self.resize_h),
                            int(self.resize_w),
                        ]

                        obj_id = len(metadata)
                        num_objs += 1
                        framenum_str = f"{frame_id:04d}_{obj_id:04d}"
                        metadata[framenum_str] = {
                            "frameId": int(frame_id),
                            "bbId": framenum_str,
                            "bbox": {
                                "x": int(object_res[0]),
                                "y": int(object_res[1]),
                                "height": int(object_res[2]),
                                "width": int(object_res[3]),
                                "object": str(object_res[4]),
                                "object_det": {
                                    "confidence": float(object_res[5]),
                                    "frameH": int(object_res[6]),
                                    "frameW": int(object_res[7]),
                                },
                            },
                        }
                    # ----------------------------------

        if "boxes" in locals():
            del boxes
        if "results" in locals():
            del results

        del frame, frame_inference
        if "cuda" in device_input:
            torch.cuda.empty_cache()
        return metadata, num_objs


# Create a dummy object representing the interface
class GPUHolder:
    def __init__(self, interface):
        self.__cuda_array_interface__ = interface


class SmartFilteringObjectDetector(BaseObjectDetector):
    def initialize(self):
        super().initialize()

        self.scale_x = self.frame_width / self.resize_w
        self.scale_y = self.frame_height / self.resize_h

        self.min_roi_w = int(self.config.ROI_MIN_AREA_RATIO * self.resize_w)
        self.min_roi_h = int(self.config.ROI_MIN_AREA_RATIO * self.resize_h)
        self.max_roi_w = int(self.resize_w * self.config.ROI_MAX_RELATIVE_SIZE_RATIO)
        self.max_roi_h = int(self.resize_h * self.config.ROI_MAX_RELATIVE_SIZE_RATIO)
        self.max_cached_elements = 100
        # Pre-allocate a single static tensor to hold the non-zero indices.
        # For a 640x640 mask, there are at most 409,600 pixels.
        # This pre-allocates a static ~3.2MB buffer in VRAM once.
        self.coords_scratchpad = torch.empty(
            (self.config.MODEL_H * self.config.MODEL_W, 2),
            dtype=torch.long,
            device="cuda",
        )

        # Pre-allocate a static box container for up to 100 candidate ROIs
        # Sized [100, 4] for [x1, y1, x2, y2]
        self.static_boxes_out = torch.empty(
            (self.max_cached_elements, 4), dtype=torch.float32, device="cuda"
        )

        # Default Kernels
        self.dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            # cv2.MORPH_RECT,
            (self.config.DILATE_KERNEL_SIZE, self.config.DILATE_KERNEL_SIZE),
        )
        # self.dilate_kernel_for_enhanced_mask = np.ones((15,15), np.uint8)  # 5, 5) (21, 21)
        self.dilate_kernel_for_enhanced_mask = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            # cv2.MORPH_RECT,
            (
                self.config.BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE,
                self.config.BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE,
            ),
        )

        # Determine minimum contour size relative to frame resolution
        self.min_contour_area = int((self.min_roi_h) * (self.min_roi_h))  # 207

        self.dist_thresh_8k = max(
            self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_width,
            self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_height,
        )
        multiplier = 2.0 if self.device_input == "cpu" else 1.0
        self.dist_thresh_640 = (
            max(
                self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_w,
                self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_h,
            )
            * multiplier
        )  # 0.05 * self.resize_w

        if not hasattr(self, "static_canvas_scratch"):
            self.static_canvas_scratch = torch.full(
                (640, 640, 3),
                fill_value=114,
                dtype=torch.uint8,
                device=self.device_input,
            )
        if self.device_input == "cuda":
            self.init_gpu_pipeline()
        else:
            self.init_cpu_pipeline()

        self.scales_tensor = torch.tensor(
            [self.scale_x, self.scale_y, self.scale_x, self.scale_y],
            device=self.device_index,
        )

        self.fixed_inference_batch = torch.empty(
            (
                self.config.MODEL_MAX_BATCH_SIZE,
                3,
                self.config.MODEL_H,
                self.config.MODEL_W,
            ),
            dtype=torch.half,
            device=self.device_input,
        )

    def run(
        self, device_frame, overall_frame_num, frame_in_clip_count=0, gt_boxes=None
    ):
        if self.device_input == "cuda":
            return self.run_gpu_pipeline(
                device_frame, overall_frame_num, frame_in_clip_count
            )
        else:
            return self.run_cpu_pipeline(
                device_frame, overall_frame_num, frame_in_clip_count
            )

    # v1
    # def get_detections_v1(
    #     self, frame, frame_id, device_input="cuda", merged=None, thickness=2
    # ):
    #     metadata = {}
    #     if merged is None or len(merged) == 0:
    #         return metadata, 0

    #     is_cuda = device_input == "cuda"
    #     num_objs = 0

    #     # FIX 1: Pre-cache class property values onto local function registers
    #     # Bypasses repeated slow overhead parsing blocks inside the hot box-loop
    #     max_batch_size = getattr(self.config, "MODEL_MAX_BATCH_SIZE", 64)
    #     target_crop_size = self.config.MODEL_W
    #     resize_w_factor = self.resize_w / self.frame_width
    #     resize_h_factor = self.resize_h / self.frame_height
    #     resize_h_int = int(self.resize_h)
    #     resize_w_int = int(self.resize_w)

    #     if is_cuda and isinstance(frame, torch.Tensor):
    #         src_tensor = frame.squeeze(0) if frame.ndim == 4 else frame
    #         if src_tensor.shape[-1] == 3:
    #             src_tensor = src_tensor.permute(2, 0, 1)
    #         src_h, src_w = src_tensor.shape[-2:]
    #     else:
    #         src_tensor = np.asarray(frame)
    #         if src_tensor.ndim == 4:
    #             src_tensor = src_tensor[0]
    #         src_h, src_w = src_tensor.shape[:2]

    #     results_pool = []
    #     patch_coordinates = []
    #     patch_idx = 0

    #     # -----------------------------------------------------------------
    #     # STEP 1: VECTORIZED SWEEP & DIRECT-WRITE MATRIX POOL
    #     # -----------------------------------------------------------------
    #     roi_patches = []
    #     for box in merged:
    #         if patch_idx >= max_batch_size:
    #             break

    #         # FIX 2: Bypassing hasattr() check loop hooks entirely
    #         # Directly extract standard bounding elements as fast local scalars
    #         box_data = box.tolist() if hasattr(box, "tolist") else box
    #         x1_raw, y1_raw, x2_raw, y2_raw = (
    #             box_data[0],
    #             box_data[1],
    #             box_data[2],
    #             box_data[3],
    #         )

    #         w_raw = x2_raw - x1_raw
    #         h_raw = y2_raw - y1_raw
    #         cx = (x1_raw + x2_raw) * 0.5
    #         cy = (y1_raw + y2_raw) * 0.5

    #         w_cushioned = w_raw + (target_crop_size * 0.1)
    #         h_cushioned = h_raw + (target_crop_size * 0.1)

    #         crop_w = max(w_cushioned, target_crop_size)
    #         crop_h = max(h_cushioned, target_crop_size)

    #         x1 = cx - (crop_w / 2.0)
    #         y1 = cy - (crop_h / 2.0)
    #         x2 = cx + (crop_w / 2.0)
    #         y2 = cy + (crop_h / 2.0)

    #         shift_left = max(0.0, 0.0 - x1)
    #         shift_right = max(0.0, x2 - src_w)
    #         x1 += shift_left - shift_right
    #         x2 += shift_left - shift_right

    #         shift_top = max(0.0, 0.0 - y1)
    #         shift_bottom = max(0.0, y2 - src_h)
    #         y1 += shift_top - shift_bottom
    #         y2 += shift_top - shift_bottom

    #         x1, y1 = max(0, int(x1)), max(0, int(y1))
    #         x2, y2 = min(src_w, int(x2)), min(src_h, int(y2))

    #         box_w, box_h = x2 - x1, y2 - y1
    #         if box_w < 8 or box_h < 8:
    #             continue

    #         scale = min(target_crop_size / box_w, target_crop_size / box_h)
    #         nw, nh = int(max(1, box_w * scale)), int(max(1, box_h * scale))
    #         dx = (target_crop_size - nw) // 2
    #         dy = (target_crop_size - nh) // 2

    #         if is_cuda and isinstance(src_tensor, torch.Tensor):
    #             # with torch.no_grad():
    #             crop = src_tensor[:, y1:y2, x1:x2]
    #             crop_resized = F.interpolate(
    #                 crop.unsqueeze(0),
    #                 size=(nh, nw),
    #                 mode="nearest",
    #             ).squeeze(0)

    #             self.fixed_inference_batch[patch_idx].fill_(114.0)
    #             self.fixed_inference_batch[patch_idx][
    #                 :, dy : dy + nh, dx : dx + nw
    #             ].copy_(crop_resized, non_blocking=True)

    #         else:
    #             crop = src_tensor[y1:y2, x1:x2]
    #             crop_resized = cv2.resize(
    #                 crop, (nw, nh), interpolation=cv2.INTER_NEAREST
    #             )
    #             padded_canvas = np.empty(
    #                 (target_crop_size, target_crop_size, 3), dtype=np.uint8
    #             )
    #             padded_canvas.fill(114)
    #             padded_canvas[dy : dy + nh, dx : dx + nw] = crop_resized

    #             self.fixed_inference_batch[patch_idx].copy_(
    #                 torch.from_numpy(padded_canvas).permute(2, 0, 1), non_blocking=True
    #             )

    #         if self.debug_frame_limit > -1:
    #             roi_patches.append(self.fixed_inference_batch[patch_idx])
    #         patch_coordinates.append((x1, y1, box_w, box_h, scale, dx, dy))
    #         patch_idx += 1

    #     if patch_idx == 0:
    #         return {}, 0

    #     if (
    #         self.debug_frame_limit > -1
    #     ):  # and self.config.DEBUG_FLAG and hasattr(self, "debug_save_crops") and len(roi_patches) > 0:
    #         self.debug_save_crops(roi_patches, frame_id)
    #         roi_patches = []

    #     # -----------------------------------------------------------------
    #     # STEP 2: INSTANT NON-BLOCKING SUB-SLICE MODEL INGESTION
    #     # -----------------------------------------------------------------
    #     # with torch.inference_mode():
    #     inference_batch = self.fixed_inference_batch[:patch_idx].clone()
    #     batch_res = self.run_model(
    #         inference_batch,
    #         imgsz=(target_crop_size, target_crop_size),
    #         batch=patch_idx,
    #         device_input=device_input,
    #         stream=STREAM_ARG,
    #     )
    #     results_pool.extend(batch_res)

    #     if is_cuda and hasattr(self, "inference_stream"):
    #         self.inference_stream.synchronize()
    #     # -----------------------------------------------------------------
    #     # OPTIMIZED STEP 3: TRUE BATCHED DEVICE HANDOFF (0ms LOCKS)
    #     # -----------------------------------------------------------------
    #     main_xyxy_list = []
    #     main_cls_list = []
    #     main_conf_list = []
    #     patch_mapping_indices = []

    #     # 1. Asynchronously collect the raw VRAM references from the model pool
    #     for idx, res in enumerate(results_pool):
    #         if res.boxes is not None and len(res.boxes) > 0:
    #             # Gather underlying torch.Tensor GPU pointer segments straight out of memory
    #             main_xyxy_list.append(res.boxes.data[:, 0:4])
    #             main_cls_list.append(res.boxes.data[:, 5])
    #             main_conf_list.append(res.boxes.data[:, 4])
    #             # Track how many objects belong to this patch index to map them back later
    #             patch_mapping_indices.append((idx, len(res.boxes)))

    #     if len(main_xyxy_list) == 0:
    #         return {}, 0

    #     # if len(main_xyxy_list) > 0:
    #     # with torch.inference_mode():
    #     # 2. Vectorized GPU Concatenation
    #     # Stacks independent slices into uniform, single-pass matrices entirely on the GPU
    #     all_main_xyxy = torch.cat(main_xyxy_list, dim=0)
    #     all_main_clss = torch.cat(main_cls_list, dim=0)
    #     all_main_confs = torch.cat(main_conf_list, dim=0)

    #     num_detected = int(all_main_xyxy.shape[0])
    #     num_clss = int(all_main_clss.shape[0])
    #     num_confs = int(all_main_confs.shape[0])

    #     if is_cuda:
    #         # Perform fast, non-blocking asynchronous streaming transfers across the PCIe bus
    #         self.pinned_cpu_xyxy[:num_detected].copy_(all_main_xyxy, non_blocking=True)
    #         cuda_int_clss = all_main_clss.to(torch.int32)
    #         self.pinned_cpu_clss[:num_clss].copy_(cuda_int_clss, non_blocking=True)
    #         self.pinned_cpu_confs[:num_confs].copy_(all_main_confs, non_blocking=True)

    #         # Synchronize ONLY the specific inference stream handle right before reading on host
    #         # self.inference_stream.synchronize()
    #         if not hasattr(self, "_dma_fence_event"):
    #             self._dma_fence_event = torch.cuda.Event()
    #         self._dma_fence_event.record(self.inference_stream)
    #         self._dma_fence_event.synchronize()

    #         all_cpu_xyxy = self.pinned_cpu_xyxy[:num_detected].numpy()
    #         all_cpu_clss = self.pinned_cpu_clss[:num_clss].numpy().flatten()
    #         all_cpu_confs = self.pinned_cpu_confs[:num_confs].numpy().flatten()
    #     else:
    #         all_cpu_xyxy = all_main_xyxy.cpu().numpy()
    #         all_cpu_clss = all_main_clss.cpu().numpy().astype(np.int32).flatten()
    #         all_cpu_confs = all_main_confs.cpu().numpy().flatten()

    #     # Global tracking increment initialization
    #     global_box_ptr = 0
    #     total_available_boxes = len(all_cpu_clss)

    #     # 4. Map detections back to global metadata layout space using our scalar index pointers
    #     for idx, num_boxes in patch_mapping_indices:
    #         ox1, oy1, box_w, box_h, scale, dx, dy = patch_coordinates[idx]

    #         for _ in range(num_boxes):
    #             if global_box_ptr >= total_available_boxes:
    #                 break

    #             lx1, ly1, lx2, ly2 = all_cpu_xyxy[global_box_ptr]
    #             class_id = int(all_cpu_clss[global_box_ptr])
    #             # if class_id < len(self.label_source):
    #             try:
    #                 if class_id >= 0 and class_id < len(self.label_source):
    #                     class_name = self.label_source[class_id]
    #                 else:
    #                     class_name = "unknown"
    #             except Exception:
    #                 traceback.print_exc()

    #             confidence = float(all_cpu_confs[global_box_ptr])

    #             lx1_unpadded = lx1 - dx
    #             ly1_unpadded = ly1 - dy
    #             lx2_unpadded = lx2 - dx
    #             ly2_unpadded = ly2 - dy

    #             global_x1 = ox1 + (lx1_unpadded / scale)
    #             global_y1 = oy1 + (ly1_unpadded / scale)
    #             global_x2 = ox1 + (lx2_unpadded / scale)
    #             global_y2 = oy1 + (ly2_unpadded / scale)

    #             disp_x = int(global_x1 * resize_w_factor)
    #             disp_y = int(global_y1 * resize_h_factor)
    #             disp_x2 = int(global_x2 * resize_w_factor)
    #             disp_y2 = int(global_y2 * resize_h_factor)

    #             disp_w = disp_x2 - disp_x
    #             disp_h = disp_y2 - disp_y

    #             if disp_w > 2 and disp_h > 2 and class_name != "unknown":
    #                 num_objs += 1
    #                 obj_id = num_objs - 1
    #                 framenum_str = f"{frame_id:04d}_{obj_id:04d}"
    #                 metadata[framenum_str] = {
    #                     "frameId": int(frame_id),
    #                     "bbId": framenum_str,
    #                     "bbox": {
    #                         "x": disp_x,
    #                         "y": disp_y,
    #                         "height": disp_h,
    #                         "width": disp_w,
    #                         "object": class_name,
    #                         "object_det": {
    #                             "confidence": confidence,
    #                             "frameH": resize_h_int,
    #                             "frameW": resize_w_int,
    #                         },
    #                     },
    #                 }

    #             global_box_ptr += 1

    #     # Clean local tensor references to protect unmanaged memory scopes
    #     crop = None
    #     del (
    #         crop,
    #         patch_coordinates,
    #         results_pool,
    #         main_xyxy_list,
    #         main_cls_list,
    #         main_conf_list,
    #     )
    #     return metadata, num_objs

    # v2
    @torch.inference_mode()
    def get_detections_v2(
        self, frame, frame_id, device_input="cuda", merged=None, thickness=2
    ):
        metadata = {}
        if merged is None or len(merged) == 0:
            return metadata, 0

        num_boxes = merged.shape[0]
        if num_boxes == 0:
            return {}, 0

        target_h, target_w = (self.config.MODEL_H, self.config.MODEL_W)

        # =========================================================================
        # STEP 1: VECTORIZED MATH FOR CUSHION & LETTERBOX PADDING
        # =========================================================================

        # 1a. Apply 10% Cushion to original boxes
        cushion = target_w * 0.1

        x1_t = (merged[:, 0] - cushion).clamp(min=0)
        y1_t = (merged[:, 1] - cushion).clamp(min=0)
        x2_t = (merged[:, 2] + cushion).clamp(max=self.frame_width)
        y2_t = (merged[:, 3] + cushion).clamp(max=self.frame_height)

        # 1b. Calculate scaled dimensions to fit within target size (preserving aspect ratio)
        box_w = (x2_t - x1_t).clamp(min=1)
        box_h = (y2_t - y1_t).clamp(min=1)

        scale = torch.min(target_w / box_w, target_h / box_h)

        new_w = (box_w * scale).int()
        new_h = (box_h * scale).int()

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # =========================================================================
        # STEP 2: HIGH-SPEED CROPPING & RESIZING LOOP
        # =========================================================================

        # Ensure frame is CHW and float for F.interpolate
        if frame.dim() == 3:
            if frame.shape[2] == 3:
                frame_chw = frame.permute(2, 0, 1).float()
            else:
                frame_chw = frame.float()
        elif frame.dim() == 4:
            frame_chw = frame.squeeze(0).float()
            if frame_chw.shape[2] == 3:
                frame_chw = frame_chw.permute(2, 0, 1)
        else:
            frame_chw = frame.float()

        # Determine correct gray padding value (114 for 0-255 range, 114/255 for 0-1 range)
        gray_val = (
            114.0 / 255.0
            if getattr(self, "normalize", False) or frame_chw.max() <= 1.0
            else 114.0
        )

        padded_batch = torch.full(
            (num_boxes, 3, target_h, target_w),
            gray_val,
            device=frame.device,
            dtype=torch.float32,
        )

        # CRITICAL PERFORMANCE FIX: Extract tensors to fast Python lists
        # This prevents the loop from halting to sync CPU/GPU on every single slice iteration
        x1_c = x1_t.int().tolist()
        y1_c = y1_t.int().tolist()
        x2_c = x2_t.int().tolist()
        y2_c = y2_t.int().tolist()
        nw_c = new_w.tolist()
        nh_c = new_h.tolist()
        px_c = pad_x.tolist()
        py_c = pad_y.tolist()

        for i in range(num_boxes):
            # 1. Slice cushioned box
            crop = frame_chw[:, y1_c[i] : y2_c[i], x1_c[i] : x2_c[i]].unsqueeze(0)

            # 2. Resize maintaining aspect ratio exactly once
            crop_resized = F.interpolate(
                crop, size=(nh_c[i], nw_c[i]), mode="bilinear", align_corners=False
            )

            # 3. Paste into the center of the padded gray canvas
            padded_batch[
                i, :, py_c[i] : py_c[i] + nh_c[i], px_c[i] : px_c[i] + nw_c[i]
            ] = crop_resized.squeeze(0)

        self.fixed_inference_batch[:num_boxes] = padded_batch.to(
            self.fixed_inference_batch.dtype
        )

        # =========================================================================
        # STEP 3: MODEL INFERENCE
        # =========================================================================
        batch_res = self.run_model(
            self.fixed_inference_batch[:num_boxes],
            imgsz=(target_h, target_w),
            batch=num_boxes,
            device_input=device_input,
            stream=STREAM_ARG,
        )

        # =========================================================================
        # STEP 4: BATCHED COORDINATE TRANSLATION
        # =========================================================================
        all_detections = []
        detection_to_crop_map = []

        for i, res in enumerate(batch_res):
            if res.boxes is not None and len(res.boxes) > 0:
                all_detections.append(res.boxes.data)
                detection_to_crop_map.extend([i] * len(res.boxes))

        if not all_detections:
            return {}, 0

        all_detections_tensor = torch.cat(all_detections, dim=0)
        detection_to_crop_map = torch.tensor(
            detection_to_crop_map, device=merged.device, dtype=torch.long
        )

        # Coordinates are local to the 640x640 padded input
        local_boxes = all_detections_tensor[:, :4]

        # Get the mapping variables for each specific detection
        scale_map = scale[detection_to_crop_map]
        pad_x_map = pad_x[detection_to_crop_map]
        pad_y_map = pad_y[detection_to_crop_map]
        orig_x1_map = x1_t[detection_to_crop_map]
        orig_y1_map = y1_t[detection_to_crop_map]

        # 4a. Remove padding offset and rescale to original cushioned crop dimensions
        unpadded_x1 = (local_boxes[:, 0] - pad_x_map) / scale_map
        unpadded_y1 = (local_boxes[:, 1] - pad_y_map) / scale_map
        unpadded_x2 = (local_boxes[:, 2] - pad_x_map) / scale_map
        unpadded_y2 = (local_boxes[:, 3] - pad_y_map) / scale_map

        # 4b. Add original cushioned crop's top-left corner to get global 8K coordinates
        global_x1 = unpadded_x1 + orig_x1_map
        global_y1 = unpadded_y1 + orig_y1_map
        global_x2 = unpadded_x2 + orig_x1_map
        global_y2 = unpadded_y2 + orig_y1_map

        # 4c. Scale from 8K to 640x640 metadata space
        meta_scale_x = self.resize_w / self.frame_width
        meta_scale_y = self.resize_h / self.frame_height

        meta_x1 = global_x1 * meta_scale_x
        meta_y1 = global_y1 * meta_scale_y
        meta_x2 = global_x2 * meta_scale_x
        meta_y2 = global_y2 * meta_scale_y

        # =========================================================================
        # STEP 5: METADATA FORMATTING
        # =========================================================================
        all_confs = all_detections_tensor[:, 4].cpu().numpy()
        all_cls_ids = all_detections_tensor[:, 5].cpu().numpy().astype(int)
        meta_boxes_cpu = (
            torch.stack([meta_x1, meta_y1, meta_x2, meta_y2], dim=1).cpu().numpy()
        )

        num_objs = 0
        for i in range(len(all_detections_tensor)):
            class_id = all_cls_ids[i]
            if class_id < 0 or class_id >= len(self.label_source):
                continue

            disp_x, disp_y, disp_x2, disp_y2 = meta_boxes_cpu[i]
            disp_w = disp_x2 - disp_x
            disp_h = disp_y2 - disp_y

            if disp_w > 2 and disp_h > 2:
                num_objs += 1
                obj_id = num_objs - 1
                framenum_str = f"{int(frame_id):04d}_{obj_id:04d}"
                metadata[framenum_str] = {
                    "frameId": int(frame_id),
                    "bbId": framenum_str,
                    "bbox": {
                        "x": int(disp_x),
                        "y": int(disp_y),
                        "height": int(disp_h),
                        "width": int(disp_w),
                        "object": self.label_source[class_id],
                        "object_det": {
                            "confidence": float(all_confs[i]),
                            "frameH": int(self.resize_h),
                            "frameW": int(self.resize_w),
                        },
                    },
                }

        del all_detections_tensor
        return metadata, num_objs

    # v3 - reduce memory footprint
    @torch.inference_mode()
    def get_detections(
        self, frame, frame_id, device_input="cuda", merged=None, thickness=2
    ):
        metadata = {}
        if merged is None or len(merged) == 0:
            return metadata, 0

        num_boxes = merged.shape[0]
        if num_boxes == 0:
            return {}, 0

        target_h, target_w = (self.config.MODEL_H, self.config.MODEL_W)

        # =========================================================================
        # STEP 1: VECTORIZED MATH FOR CUSHION & LETTERBOX PADDING
        # =========================================================================

        # 1a. Apply 10% Cushion to original boxes
        cushion = target_w * 0.1

        x1_t = (merged[:, 0] - cushion).clamp(min=0).detach().cpu()
        y1_t = (merged[:, 1] - cushion).clamp(min=0).detach().cpu()
        x2_t = (merged[:, 2] + cushion).clamp(max=self.frame_width).detach().cpu()
        y2_t = (merged[:, 3] + cushion).clamp(max=self.frame_height).detach().cpu()

        # 1b. Calculate scaled dimensions to fit within target size (preserving aspect ratio)
        box_w = (x2_t - x1_t).clamp(min=1).detach().cpu()
        box_h = (y2_t - y1_t).clamp(min=1).detach().cpu()

        scale = torch.min(target_w / box_w, target_h / box_h).detach().cpu()

        new_w = (box_w * scale).int()
        new_h = (box_h * scale).int()

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # =========================================================================
        # STEP 2: HIGH-SPEED CROPPING & RESIZING LOOP
        # =========================================================================

        # Ensure frame is CHW and float for F.interpolate
        if frame.dim() == 3:
            if frame.shape[2] == 3:
                # frame_chw = frame.permute(2, 0, 1).float()
                # Permute is just a "view" metadata change (0ms, 0 VRAM allocation)
                permuted_view = frame.permute(2, 0, 1)

                # Copy and cast directly into our pre-allocated static float buffer in-place!
                # This allocates 0 MB of new memory!
                self.gpu_float_staging.copy_(permuted_view, non_blocking=True)

                # Now divide by 255.0 in-place to normalize (reuses the exact same float32 memory)
                frame_chw = self.gpu_float_staging[0]  # .div_(255.0)
            else:
                frame_chw = frame.float()
        elif frame.dim() == 4:
            frame_chw = frame.squeeze(0).float()
            if frame_chw.shape[2] == 3:
                frame_chw = frame_chw.permute(2, 0, 1)
        else:
            frame_chw = frame.float()

        # Determine correct gray padding value (114 for 0-255 range, 114/255 for 0-1 range)
        gray_val = (
            114.0 / 255.0
            if getattr(self, "normalize", False) or frame_chw.max() <= 1.0
            else 114.0
        )

        # padded_batch = torch.full(
        #     (num_boxes, 3, target_h, target_w),
        #     gray_val,
        #     device=frame.device,
        #     dtype=torch.float32,
        # )

        # CRITICAL PERFORMANCE FIX: Extract tensors to fast Python lists
        # This prevents the loop from halting to sync CPU/GPU on every single slice iteration
        x1_c = x1_t.int().tolist()
        y1_c = y1_t.int().tolist()
        x2_c = x2_t.int().tolist()
        y2_c = y2_t.int().tolist()
        nw_c = new_w.tolist()
        nh_c = new_h.tolist()
        px_c = pad_x.tolist()
        py_c = pad_y.tolist()

        for i in range(num_boxes):
            # 1. Slice cushioned box
            crop = frame_chw[:, y1_c[i] : y2_c[i], x1_c[i] : x2_c[i]].unsqueeze(0)

            # 2. Resize maintaining aspect ratio exactly once
            crop_resized = F.interpolate(
                crop, size=(nh_c[i], nw_c[i]), mode="bilinear", align_corners=False
            )

            # 3. Paste into the center of the padded gray canvas
            # padded_batch[
            #     i, :, py_c[i] : py_c[i] + nh_c[i], px_c[i] : px_c[i] + nw_c[i]
            # ] = crop_resized.squeeze(0)
            self.fixed_inference_batch[
                i, :, py_c[i] : py_c[i] + nh_c[i], px_c[i] : px_c[i] + nw_c[i]
            ] = crop_resized.squeeze(0)

        # self.fixed_inference_batch[:num_boxes] = padded_batch.to(
        #     self.fixed_inference_batch.dtype
        # )

        # =========================================================================
        # STEP 3: MODEL INFERENCE
        # =========================================================================

        # =========================================================================
        # STEP 4: BATCHED COORDINATE TRANSLATION
        # =========================================================================
        # all_detections = []
        # detection_to_crop_map = []
        num_objs = 0
        meta_scale_x = self.resize_w / self.frame_width
        meta_scale_y = self.resize_h / self.frame_height

        with torch.no_grad():
            batch_res = self.run_model(
                self.fixed_inference_batch[:num_boxes],
                imgsz=(target_h, target_w),
                batch=num_boxes,
                device_input=device_input,
                stream=STREAM_ARG,
            )

            for i, res in enumerate(batch_res):
                if res.boxes is None or len(res.boxes) == 0:
                    del res
                    continue

                # Move data to CPU immediately to free VRAM
                local_boxes = res.boxes.data.detach().cpu()

                # Get the mapping variables for each specific detection
                scale_map = scale[i]
                pad_x_map = pad_x[i]
                pad_y_map = pad_y[i]
                orig_x1_map = x1_t[i]
                orig_y1_map = y1_t[i]

                # 4a. Remove padding offset and rescale to original cushioned crop dimensions
                unpadded_x1 = (local_boxes[:, 0] - pad_x_map) / scale_map
                unpadded_y1 = (local_boxes[:, 1] - pad_y_map) / scale_map
                unpadded_x2 = (local_boxes[:, 2] - pad_x_map) / scale_map
                unpadded_y2 = (local_boxes[:, 3] - pad_y_map) / scale_map

                # 4b. Add original cushioned crop's top-left corner to get global 8K coordinates
                global_x1 = unpadded_x1 + orig_x1_map
                global_y1 = unpadded_y1 + orig_y1_map
                # global_x2 = unpadded_x2 + orig_x1_map
                # global_y2 = unpadded_y2 + orig_y1_map

                # 4c. Scale from 8K to 640x640 metadata space
                meta_x1 = global_x1 * meta_scale_x
                meta_y1 = global_y1 * meta_scale_y
                # meta_x2 = global_x2 * meta_scale_x
                # meta_y2 = global_y2 * meta_scale_y
                meta_w = (unpadded_x2 - unpadded_x1) * meta_scale_x
                meta_h = (unpadded_y2 - unpadded_y1) * meta_scale_y

                # =========================================================================
                # STEP 5: METADATA FORMATTING
                # =========================================================================
                # Format metadata
                for j in range(len(local_boxes)):
                    if meta_w[j] > 2 and meta_h[j] > 2:
                        class_id = int(local_boxes[j, 5])
                        if 0 <= class_id < len(self.label_source):
                            framenum_str = f"{int(frame_id):04d}_{num_objs:04d}"
                            metadata[framenum_str] = {
                                "frameId": int(frame_id),
                                "bbId": framenum_str,
                                "bbox": {
                                    "x": int(meta_x1[j]),
                                    "y": int(meta_y1[j]),
                                    "height": int(meta_h[j]),
                                    "width": int(meta_w[j]),
                                    "object": self.label_source[class_id],
                                    "object_det": {
                                        "confidence": float(local_boxes[j, 4]),
                                        "frameH": int(self.resize_h),
                                        "frameW": int(self.resize_w),
                                    },
                                },
                            }
                            num_objs += 1

                # Explicitly delete tensors to free memory inside the loop
                del res, local_boxes

            del batch_res
            self.fixed_inference_batch.fill_(gray_val)

        if "cuda" in device_input:
            torch.cuda.empty_cache()
        return metadata, num_objs

    # GPU ------------------------------------------------

    def init_gpu_pipeline(self):
        self.gpu_id = 0
        self.inference_stream = torch.cuda.Stream()

        if self.timer_enabled:
            self.sf_start, self.sf_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.roi_start, self.roi_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.det_start, self.det_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        # --- START: FUSED KERNEL COMPILATION ---
        # This CUDA C++ kernel fuses a 3x3 Box Blur and a binary threshold (value: 50).
        # A fused dilation is very complex; we apply it separately for correctness.
        # This still reduces 3 kernel launches (Blur, Thresh, Dilate) to 2 (Fused, Dilate).
        fused_blur_thresh_kernel_code = r"""
        extern "C" __global__
        void fused_blur_thresh(const unsigned char* src, unsigned char* dst, int width, int height, int src_step, int dst_step) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            // 1. Fused 3x3 Box Blur
            float sum = 0.0f;
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += src[ny * src_step + nx];
                        count++;
                    }
                }
            }
            float blurred_val = sum / count;

            // 2. Fused Threshold
            unsigned char output_val = (blurred_val > 50.0f) ? 255 : 0;

            // Write the final result
            dst[y * dst_step + x] = output_val;
        }
        """
        self.fused_blur_thresh_kernel = cupy.RawKernel(
            fused_blur_thresh_kernel_code, "fused_blur_thresh"
        )

        # This CUDA C++ kernel fuses a 3x3 Box Blur, a binary threshold (50), and a 3x3 Dilation.
        # fused_3_in_1_kernel_code = r'''
        # #define TILE_DIM 16
        # #define BLOCK_DIM 16

        # extern "C" __global__
        # void fused_btd(const unsigned char* src, unsigned char* dst, int width, int height, int src_step, int dst_step) {

        #     // Shared memory for the source tile and the intermediate thresholded tile
        #     __shared__ unsigned char s_src_tile[TILE_DIM + 2][TILE_DIM + 2];
        #     __shared__ unsigned char s_thresh_tile[TILE_DIM][TILE_DIM];

        #     int tx = threadIdx.x;
        #     int ty = threadIdx.y;
        #     int gx = blockIdx.x * TILE_DIM + tx;
        #     int gy = blockIdx.y * TILE_DIM + ty;

        #     // Load source data into shared memory (including a 1-pixel halo)
        #     for (int i = ty; i < TILE_DIM + 2; i += BLOCK_DIM) {
        #         for (int j = tx; j < TILE_DIM + 2; j += BLOCK_DIM) {
        #             int load_x = gx - 1 + j;
        #             int load_y = gy - 1 + i;
        #             if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
        #                 s_src_tile[i][j] = src[load_y * src_step + load_x];
        #             } else {
        #                 s_src_tile[i][j] = 0;
        #             }
        #         }
        #     }
        #     __syncthreads();

        #     // --- 1. Fused Blur & Threshold ---
        #     if (tx < TILE_DIM && ty < TILE_DIM) {
        #         float sum = 0.0f;
        #         for (int dy = 0; dy <= 2; ++dy) {
        #             for (int dx = 0; dx <= 2; ++dx) {
        #                 sum += s_src_tile[ty + dy][tx + dx];
        #             }
        #         }
        #         float blurred_val = sum / 9.0f;
        #         s_thresh_tile[ty][tx] = (blurred_val > 50.0f) ? 255 : 0;
        #     }
        #     __syncthreads();

        #     // --- 2. Fused Dilation ---
        #     if (tx < TILE_DIM && ty < TILE_DIM && gx < width && gy < height) {
        #         unsigned char max_val = 0;
        #         for (int dy = -1; dy <= 1; ++dy) {
        #             for (int dx = -1; dx <= 1; ++dx) {
        #                 int check_x = tx + dx;
        #                 int check_y = ty + dy;
        #                 if (check_x >= 0 && check_x < TILE_DIM && check_y >= 0 && check_y < TILE_DIM) {
        #                     if (s_thresh_tile[check_y][check_x] > max_val) {
        #                         max_val = s_thresh_tile[check_y][check_x];
        #                     }
        #                 }
        #             }
        #         }
        #         dst[gy * dst_step + gx] = max_val;
        #     }
        # }
        # '''

        # fused_3_in_1_kernel_code = r'''
        # extern "C" __global__
        # void fused_btd(const unsigned char* src, unsigned char* dst, int width, int height, int src_step, int dst_step) {
        #     int x = blockIdx.x * blockDim.x + threadIdx.x;
        #     int y = blockIdx.y * blockDim.y + threadIdx.y;

        #     if (x >= width || y >= height) return;

        #     // --- STEP 1 & 2: 3x3 NEIGHBORHOOD BLUR & THRESHOLD ---
        #     // We compute the thresholded values of the local 3x3 neighborhood on the fly
        #     unsigned char local_thresh[3][3];

        #     for (int dy = -1; dy <= 1; ++dy) {
        #         for (int dx = -1; dx <= 1; ++dx) {
        #             int nx = x + dx;
        #             int ny = y + dy;

        #             // Clamp coordinates to image edges safely
        #             nx = (nx < 0) ? 0 : ((nx >= width) ? width - 1 : nx);
        #             ny = (ny < 0) ? 0 : ((ny >= height) ? height - 1 : ny);

        #             // Compute blur for coordinate (nx, ny)
        #             float sum = 0.0f;
        #             int count = 0;
        #             for (int k_dy = -1; k_dy <= 1; ++k_dy) {
        #                 for (int k_dx = -1; k_dx <= 1; ++k_dx) {
        #                     int k_nx = nx + k_dx;
        #                     int k_ny = ny + k_dy;

        #                     k_nx = (k_nx < 0) ? 0 : ((k_nx >= width) ? width - 1 : k_nx);
        #                     k_ny = (k_ny < 0) ? 0 : ((k_ny >= height) ? height - 1 : k_ny);

        #                     sum += src[k_ny * src_step + k_nx];
        #                     count++;
        #                 }
        #             }
        #             float blurred_val = sum / (float)count;

        #             // Threshold instantly
        #             local_thresh[dy + 1][dx + 1] = (blurred_val > 50.0f) ? 255 : 0;
        #         }
        #     }

        #     // --- STEP 3: DILATE ON THRESHOLDED RESULTS ---
        #     unsigned char max_val = 0;
        #     for (int dy = 0; dy < 3; ++dy) {
        #         for (int dx = 0; dx < 3; ++dx) {
        #             if (local_thresh[dy][dx] > max_val) {
        #                 max_val = local_thresh[dy][dx];
        #             }
        #         }
        #     }

        #     // Write the final dilate value safely to global memory
        #     dst[y * dst_step + x] = max_val;
        # }
        # '''
        fused_3_in_1_kernel_code = r"""
        extern "C" __global__
        void fused_btd(const unsigned char* src, unsigned char* dst, int width, int height, int src_step, int dst_step) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            // --- STEP 1 & 2: 17x17 BLUR & THRESHOLD ---
            // 5x5 local neighborhood for the Dilation phase
            unsigned char local_thresh[5][5];

            // 5x5 Dilation radius (dy, dx from -2 to 2)
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    // Clamp coordinates to image edges safely
                    nx = (nx < 0) ? 0 : ((nx >= width) ? width - 1 : nx);
                    ny = (ny < 0) ? 0 : ((ny >= height) ? height - 1 : ny);

                    // 5x5 Blur radius (k_dy, k_dx from -8 to 8 (for 17x17))
                    float sum = 0.0f;
                    int count = 0;
                    for (int k_dy = -2; k_dy <= 2; ++k_dy) {
                        for (int k_dx = -8; k_dx <= 8; ++k_dx) {
                            int k_nx = nx + k_dx;
                            int k_ny = ny + k_dy;

                            k_nx = (k_nx < 0) ? 0 : ((k_nx >= width) ? width - 1 : k_nx);
                            k_ny = (k_ny < 0) ? 0 : ((k_ny >= height) ? height - 1 : k_ny);

                            sum += src[k_ny * src_step + k_nx];
                            count++;
                        }
                    }
                    float blurred_val = sum / (float)count;

                    // Threshold instantly (offset dx, dy by +2 to fit in 0-4 array indices)
                    local_thresh[dy + 2][dx + 2] = (blurred_val > 50.0f) ? 255 : 0;
                }
            }

            // --- STEP 3: 5x5 DILATE ON THRESHOLDED RESULTS ---
            unsigned char max_val = 0;
            for (int dy = 0; dy < 5; ++dy) {
                for (int dx = 0; dx < 5; ++dx) {
                    if (local_thresh[dy][dx] > max_val) {
                        max_val = local_thresh[dy][dx];
                    }
                }
            }

            // Write the final dilate value safely to global memory
            dst[y * dst_step + x] = max_val;
        }
        """

        self.fused_3_in_1_kernel = cupy.RawKernel(fused_3_in_1_kernel_code, "fused_btd")
        # --- END: FUSED KERNEL COMPILATION ---

        self.prepare_gpu_pipeline()

    def allocate_gpu(self):
        """
        Allocates persistent GpuMat buffers and CUDA streams to
        enable zero-copy GPU processing.
        """
        self.device_index = f"cuda:{self.gpu_id}"

        # Safely extract the primitive compiled C++ function pointer out of CuPy's RawKernel
        if hasattr(get_bounds_kernel, "kernel"):
            self._raw_bounds_function = get_bounds_kernel.kernel
        else:
            # Fallback handle if utilizing an alternate CuPy execution wrapper mapping
            self._raw_bounds_function = get_bounds_kernel

        ksize = (17, 17)
        self._cuda_gaussian_filter = cv2.cuda.createGaussianFilter(
            srcType=cv2.CV_8UC1, dstType=cv2.CV_8UC1, ksize=ksize, sigma1=0
        )

        self.recycled_resize_mat = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC3
        )
        # Pre-allocate double buffers for resizing
        # self.recycled_resize_mat_A = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)
        # self.recycled_resize_mat_B = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)

        # Track which buffer is active
        self.use_buffer_A = True

        self.raw_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.thresh_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.clean_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.d_blurred = cv2.cuda.GpuMat(self.clean_mask.size(), cv2.CV_8UC1)

        # Compile the kernel once at startup
        # self._row_bounds_function = cupy.RawKernel(PROPAGATION_KERNEL_CODE, "get_row_bounds_fused")

        # self.fixed_inference_batch = torch.empty(
        #     (
        #         self.config.MODEL_MAX_BATCH_SIZE,
        #         3,
        #         self.config.MODEL_H,
        #         self.config.MODEL_W,
        #     ),
        #     dtype=torch.half,
        #     device="cuda",
        # )

        # self.gpu_float_staging = None

        self.gpu_float_staging = torch.empty(
            (1, 3, self.frame_height, self.frame_width),
            dtype=torch.float16,
            device=self.device_index,
        )
        # self.stream = cv2.cuda.Stream()
        # self.ingest_stream = torch.cuda.Stream()
        # self.inference_stream = torch.cuda.Stream()
        self.bgs_stream = cv2.cuda.Stream()
        # self.gpu_fullres_frame = cv2.cuda.GpuMat(
        #     self.frame_height, self.frame_width, cv2.CV_8UC3
        # )
        # self.resized_gpumat = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)
        # self.resized_frame = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)
        # self.resized_frame.setTo(0, self.bgs_stream)
        # self.fgMask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.prev_bkgd = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        if self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD == "and":
            self.prev_bkgd.setTo((1,))
        else:
            self.prev_bkgd.setTo((0,))
        self.mask_history = deque(
            maxlen=self.config.BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE
        )
        self.mask_history.append(self.prev_bkgd)

        self.gpu_threshold_dst_frame = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        self.gpu_morphed_frame = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )

        self.upload_stream = cv2.cuda.Stream()
        self.upload_event = cv2.cuda.Event()

        # self.queue_capacity = int(2 * self.target_fps)  # 60
        self.num_buffers = int(2 * self.target_fps)  # self.queue_capacity + 5
        self.gpu_buffer_pool = [
            cv2.cuda.GpuMat(self.frame_height, self.frame_width, cv2.CV_8UC3)
            for _ in range(self.num_buffers)
        ]
        self.buffer_idx = 0

        self.frame_buffer_pool = [
            torch.empty(
                (3, self.frame_height, self.frame_width),
                dtype=torch.uint8,
                device="cuda",
            )
            for _ in range(2)
        ]
        self.pool_idx = 0

        # Create a matching pool of pinned host memory for the 8K frames
        # self.host_buffer_pool = [
        #     cv2.cuda.HostMem(self.frame_height, self.frame_width, cv2.CV_8UC3)
        #     for _ in range(self.num_buffers)
        # ]

        # Create continuous buffers to prevent stride artifacts during 8K downloads
        self.pinned_downloaded_resizedframe_np = cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC3
        )
        self.pinned_downloaded_frame_np = cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_threshold_dst_frame
        )
        cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_morphed_frame
        )

        # This prevents the AI thread from overwriting the encoder's data.
        self.gpu_encoder_8k_buf = cv2.cuda.createContinuous(
            self.frame_height, self.frame_width, cv2.CV_8UC3
        )

        # Continuous allocation prevents stride/padding artifacts
        self.gpu_display_frame = cv2.cuda.createContinuous(
            self.disp_h, self.disp_w, cv2.CV_8UC3
        )

        # Create a dedicated background stream for encoding tasks
        self.encode_stream = cv2.cuda.Stream()

        # Allocate a permanent float32/float16 channel layout space directly on VRAM
        # self.static_gpu_360p = torch.empty(
        #     (1, 3, self.disp_h, self.disp_w),
        #     dtype=torch.float32,
        #     device="cuda",
        # )
        # self.static_gpu_byte_bchw = torch.empty(
        #     (1, 3, self.disp_h, self.disp_w),
        #     dtype=torch.uint8,
        #     device="cuda",
        # ).contiguous()

        # Allocates a fixed memory space directly accessible by your GPU DMA engine
        self.pinned_cpu_xyxy = torch.empty(
            (self.config.MAX_DETECTIONS, 4), dtype=torch.float32, device="cpu"
        ).pin_memory()
        self.pinned_cpu_clss = torch.empty(
            (self.config.MAX_DETECTIONS,), dtype=torch.int32, device="cpu"
        ).pin_memory()
        self.pinned_cpu_confs = torch.empty(
            (self.config.MAX_DETECTIONS,), dtype=torch.float32, device="cpu"
        ).pin_memory()

        self.cupy_structure = cupy.ones((3, 3), dtype=cupy.int32)  # , order="C")
        self._max_labels = 1024  # Cap maximum tracking elements per frame
        # Pre-allocate static, persistent workspace array caches straight in VRAM
        self._x1_pool = cupy.zeros((self._max_labels,), dtype=cupy.int32)
        self._y1_pool = cupy.zeros((self._max_labels,), dtype=cupy.int32)
        self._x2_pool = cupy.zeros((self._max_labels,), dtype=cupy.int32)
        self._y2_pool = cupy.zeros((self._max_labels,), dtype=cupy.int32)

        # Pre-allocate the labeled output tensor space to avoid internal allocation churn
        # Match your target resized canvas dimensions (e.g., 640x640)
        self._labeled_scratch = cupy.empty(
            (self.resize_h, self.resize_w), dtype=cupy.int32, order="C"
        )

        # Create two isolated tracking canvases to handle the ping-pong data stream
        # self.static_host_canvases = [
        #     np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8),
        #     np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8),
        # ]
        # self.canvas_selector = 0

        # # Register BOTH buffers as page-locked memory
        # cv2.cuda.registerPageLocked(self.static_host_canvases[0])
        # cv2.cuda.registerPageLocked(self.static_host_canvases[1])

    def prepare_gpu_pipeline(self):
        self.allocate_gpu()

        # Subtraction
        # self.backSub_lock = threading.Lock()
        # history = int(2 * self.target_fps)  # 300  # int(5 * self.target_fps)
        self.lr = self.config.BKGD_SUB_MOG2_LR  # 1 / history
        self.backSub = cv2.cuda.createBackgroundSubtractorMOG2(
            history=self.config.BKGD_SUB_MOG2_HISTORY,  # Clear ghosts of fast drones in ~2 seconds (2*fps)
            varThreshold=int(
                self.config.BKGD_SUB_MOG2_VARTHRESHOLD  # 1.15
            ),  # High threshold to ignore "shimmer" and compression noise  # default 16
            # varThreshold=50,  #self.config.BKGD_SUB_MOG2_VARTHRESHOLD,  # High threshold to ignore "shimmer" and compression noise  # default 16
            # CUDA implementation of MOG2 often requires a higher varThreshold to achieve the same "cleanliness" as the CPU (15-20%)
            detectShadows=self.config.BKGD_SUB_MOG2_DETECTSHADOWS,  # default True
        )
        self.opencv_bgs_output = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        self.opencv_bgs_dilate_output = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )

        # Force the GPU to match the CPU's background criteria limits
        # self.backSub.setBackgroundRatio(0.05)                 # Standardize background matching speed
        # self.backSub.setComplexityReductionThreshold(0.05)     # Drop unstable low-variance noise regions
        # self.backSub.setVarMin(4.0)                           # High-pass filter to erase micro-vibrations

        self.dilate_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8U, self.dilate_kernel
        )
        self.dilate_filter_for_enhanced_mask = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8UC1, self.dilate_kernel_for_enhanced_mask
        )
        # # self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # self.morph_filter = cv2.cuda.createMorphologyFilter(
        #     cv2.MORPH_DILATE, cv2.CV_8UC1, self.morph_kernel
        # )
        # self.labels_gpu = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_32S)
        # self.labels_gpu = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8U)
        # self.labels_gpu.setTo(0, self.bgs_stream)

    def gpu_warmup(self):
        """
        Comprehensive, non-blocking GPU warmup harness.
        Pre-compiles CuPy Scipy labeling kernels, box-merging logic,
        and TensorRT engine profiles without disk writing side effects.
        """
        # import cv2
        # import torch

        self.warmup_stream = cv2.cuda.Stream()

        h, w = self.resize_h, self.resize_w

        # 1. Standard OpenCV CUDA Pre-processing Compilation
        gpu_warmup_input_frame = cv2.cuda_GpuMat(h, w, cv2.CV_8U)
        if gpu_warmup_input_frame is not None:
            gpu_warmup_input_frame.setTo(255)
            gpu_warmup_frame = cv2.cuda_GpuMat(h, w, cv2.CV_8U)

            cv2.cuda.resize(
                gpu_warmup_input_frame,
                (w, h),
                stream=self.warmup_stream,
                dst=gpu_warmup_frame,
                interpolation=cv2.INTER_NEAREST,
            )

            gpu_threshold_dst_frame = cv2.cuda_GpuMat(h, w, cv2.CV_8U)
            cv2.cuda.threshold(
                gpu_warmup_frame,
                self.config.THRESHOLD_VALUE,
                self.config.THRESHOLD_MAX_VALUE,
                cv2.THRESH_BINARY,
                gpu_threshold_dst_frame,
                self.warmup_stream,
            )

            gpu_morphed_frame = cv2.cuda_GpuMat(h, w, cv2.CV_8U)
            self.dilate_filter.apply(
                gpu_threshold_dst_frame, gpu_morphed_frame, self.warmup_stream
            )
        active_stream_ptr = (
            self.bgs_stream.cudaPtr() if hasattr(self, "bgs_stream") else 0
        )
        self.warmup_stream.waitForCompletion()

        mock_mask = cupy.zeros((640, 640), dtype=cupy.uint8)
        mock_mask[100:150, 100:150] = 1  # Draw a dummy blob
        # mock_scratch = cupy.empty((640, 640), dtype=cupy.int32)
        # mock_structure = cupy.ones((3, 3), dtype=cupy.int32)

        # Invoke once to trigger NVRTC compilation and cache the binary module to disk
        # cupyx.scipy.ndimage.label(mock_mask, structure=self.cupy_structure, output=mock_scratch)
        with cupy.cuda.ExternalStream(active_stream_ptr):
            cupyx.scipy.ndimage.label(
                mock_mask, structure=self.cupy_structure, output=self._labeled_scratch
            )
        # cupy.cuda.stream.get_current_stream().synchronize()
        cupy.cuda.ExternalStream(active_stream_ptr).synchronize()
        main_app_logger.info("[WARMUP] CuPy SciPy Labeling kernels fully cached.")

        # 🚀 2. PIPELINE EXTENSION: Pre-allocate static canvas memories if missing
        # if not hasattr(self, "static_canvas_scratch"):
        #     self.static_canvas_scratch = torch.full(
        #         (640, 640, 3), fill_value=114, dtype=torch.uint8, device="cuda"
        #     )

        # 🚀 3. PIPELINE EXTENSION: Pre-compile AI Model and Box reduction pipelines
        # Capture the original debug configuration state
        original_debug_flag = getattr(self.config, "DEBUG_FLAG", False)

        try:
            # Force the debug flag off during warmup loop to completely block debug_save_crops
            self.config.DEBUG_FLAG = False

            # Construct a synthetic 8K model input canvas block on the GPU device
            # Adjust dimensions if your raw source video maps differently than [8192, 8192, 3]
            synthetic_8k_tensor = torch.zeros(
                (self.frame_height, self.frame_width, 3),
                dtype=torch.uint8,
                device="cuda",
            )

            # Create a standard mock overlapping region block to exercise merge_boxes_gpu
            # Shapes: [N, 4] -> format (x1, y1, x2, y2)
            mock_overlapping_boxes = torch.tensor(
                [
                    [100.0, 100.0, 250.0, 250.0],
                    [102.0, 101.0, 248.0, 252.0],  # Overlapping twin bounding box
                    [500.0, 500.0, 640.0, 640.0],
                ],
                dtype=torch.float32,
                device="cuda",
            )

            # Run exactly 3 dummy steps to guarantee CuPy Scipy features and TensorRT allocations cache
            main_app_logger.info(
                "[WARMUP] Priming algorithm pipeline layers asynchronously..."
            )
            for _ in range(3):
                with torch.inference_mode():
                    # Feed through the master detection utility function directly
                    # _, _ = self.get_detections_with_smart_filtering(
                    #     frame=synthetic_8k_tensor,
                    #     frame_id=0,
                    #     device_input="cuda",
                    #     merged=mock_overlapping_boxes,
                    # )
                    self.get_detections(
                        frame=synthetic_8k_tensor,
                        frame_id=0,
                        device_input="cuda",
                        merged=mock_overlapping_boxes
                        if self.config.sf_enabled
                        else None,
                    )

            # Hard fence sync to secure compilation layers across active CUDA contexts
            # torch.cuda.synchronize()
            main_app_logger.info(
                "[WARMUP] Complete. All pipeline pathways completely compiled."
            )

        finally:
            # Safely restore your original testing workflow debug constraints
            self.config.DEBUG_FLAG = original_debug_flag

        # Release resources
        del synthetic_8k_tensor, mock_overlapping_boxes
        del (
            gpu_warmup_input_frame,
            gpu_warmup_frame,
            gpu_threshold_dst_frame,
            gpu_morphed_frame,
        )
        if hasattr(self, "warmup_stream"):
            delattr(self, "warmup_stream")

        torch.cuda.empty_cache()

    def cleanup_gpu_v1(self):
        """
        Explicitly releases all GPU-allocated memory to prevent
        VRAM leaks in 8K concurrent streams.
        """
        # Iterate through class attributes to explicitly release VRAM.
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)

            # Check if the attribute is a GpuMat
            if isinstance(attr_value, cv2.cuda.GpuMat):
                # 🏎️ Force the NVIDIA driver to deallocate this specific memory segment
                attr_value.release()
                setattr(self, attr_name, None)
                main_app_logger.info(f"✅ Released GpuMat: {attr_name}")

        # if hasattr(self, "gpu_fullres_frame") and self.gpu_fullres_frame is not None:
        #     try:
        #         self.gpu_fullres_frame.release()
        #     except Exception:
        #         self.gpu_fullres_frame = None

        if hasattr(self, "gpu_morphed_frame") and self.gpu_morphed_frame is not None:
            try:
                self.gpu_morphed_frame.release()
            except Exception:
                self.gpu_morphed_frame = None

        # if hasattr(self, "labels_gpu") and self.labels_gpu is not None:
        #     try:
        #         self.labels_gpu.release()
        #     except Exception:
        #         self.labels_gpu = None

        if hasattr(self, "gpu_encoder_8k_buf") and self.gpu_encoder_8k_buf is not None:
            try:
                self.gpu_encoder_8k_buf.release()
            except Exception:
                self.gpu_encoder_8k_buf = None

        if hasattr(self, "gpu_display_frame") and self.gpu_display_frame is not None:
            try:
                self.gpu_display_frame.release()
            except Exception:
                self.gpu_display_frame = None

        if hasattr(self, "gpu_crop_batch"):
            for mat in self.gpu_crop_batch:
                if isinstance(mat, cv2.cuda.GpuMat):
                    mat.release()
            self.gpu_crop_batch = []

        # if hasattr(self, "stream"):
        #     self.stream.waitForCompletion()

        self.pinned_downloaded_resizedframe_np = None
        self.gpu_threshold_dst_frame = None
        self.gpu_morphed_frame = None
        self.pinned_downloaded_frame_np = None

        # Handle specific buffers (like your Ping-Pong lists)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

        # Optional: Final flush of the CUDA caching allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_background_subtraction_gpu(
        self, gpu_frame, include_history=True, method="and", stream=None
    ):
        stream = stream if isinstance(stream, cv2.cuda.Stream) else self.bgs_stream

        # raw_mask = self.backSub.apply(
        #     gpu_frame,
        #     # motion_input,
        #     float(self.lr),  # 0.005,  # float(self.lr),
        #     stream=stream,
        # )
        self.backSub.apply(
            image=gpu_frame,
            fgmask=self.opencv_bgs_output,
            learningRate=float(self.lr),
            stream=stream,
        )
        raw_mask = self.opencv_bgs_output

        if include_history:
            # if self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD == "and":
            #     self.prev_bkgd.setTo((1,))
            # else:
            #     self.prev_bkgd.setTo((0,))
            # If the deque is full, pop the oldest GpuMat and explicitly release it.
            if len(self.mask_history) >= self.mask_history.maxlen:
                old_mask = self.mask_history.popleft()
                if old_mask is not None:
                    # It releases the VRAM held by the oldest mask.
                    old_mask.release()

            for m in list(self.mask_history):
                # Dilate the historical mask on GPU
                # dilated = self.dilate_filter_for_enhanced_mask.apply(m, stream=stream)
                self.dilate_filter_for_enhanced_mask.apply(
                    src=m, dst=self.opencv_bgs_dilate_output, stream=stream
                )

                if method == "or":
                    # Bitwise OR on GPU
                    cv2.cuda.bitwise_or(
                        self.prev_bkgd,
                        self.opencv_bgs_dilate_output,
                        self.prev_bkgd,
                        stream=stream,
                    )
                else:
                    # Bitwise AND on GPU
                    cv2.cuda.bitwise_and(
                        self.prev_bkgd,
                        self.opencv_bgs_dilate_output,
                        self.prev_bkgd,
                        stream=stream,
                    )

            self.mask_history.append(raw_mask.clone())  # .clone()

            raw_mask = cv2.cuda.bitwise_or(
                raw_mask, self.prev_bkgd, stream=self.bgs_stream
            )
        return raw_mask

    def get_sf_gpu_rois_v1(
        self, device_frame, overall_frame_num, max_candidates=100, limit_640=1280
    ):
        debug_frame_limit = self.debug_frame_limit

        if overall_frame_num <= debug_frame_limit:
            stage_debug_dir = (
                self.result_dir / "debug_stages" / self._testMethodName / "roi_stages"
            )
            stage_debug_dir.mkdir(parents=True, exist_ok=True)

        # 2. BRIDGE THE PYTORCH TO OPENCV VRAM GAP
        if torch.is_tensor(device_frame):
            h_raw, w_raw, ch = device_frame.shape
            cuda_mem_ptr = device_frame.data_ptr()
            cv_type = cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1
            row_step_bytes = device_frame.stride()[0] * device_frame.element_size()
            src_gpu_mat = cv2.cuda.createGpuMatFromCudaMemory(
                h_raw, w_raw, cv_type, cuda_mem_ptr, step=row_step_bytes
            )
        else:
            src_gpu_mat = device_frame
            ch = src_gpu_mat.channels()

        if overall_frame_num <= debug_frame_limit:
            src_cpu = src_gpu_mat.download()
            if ch == 3:
                src_cpu = cv2.cvtColor(src_cpu, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{overall_frame_num:04d}_stage1_src.jpg"),
                src_cpu,
            )

        # 3. STRIDED LAYOUT INITIALIZATION
        # if not hasattr(self, "raw_mask"):
        #     self.recycled_resize_mat = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1)
        #     self.raw_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        #     self.thresh_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        #     self.clean_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        #     self.d_blurred = cv2.cuda.GpuMat(self.clean_mask.size(), cv2.CV_8UC1)

        # 4. RUN ASYNCHRONOUS DOWN-SAMPLING GATE
        cv2.cuda.resize(
            src_gpu_mat,
            dst=self.recycled_resize_mat,
            dsize=(self.resize_w, self.resize_h),
            interpolation=cv2.INTER_NEAREST,
            stream=self.bgs_stream,
        )

        if overall_frame_num <= debug_frame_limit:
            self.bgs_stream.waitForCompletion()
            resize_cpu = self.recycled_resize_mat.download()
            if ch == 3:
                resize_cpu = cv2.cvtColor(resize_cpu, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(
                    stage_debug_dir / f"frame_{overall_frame_num:04d}_stage2_resize.jpg"
                ),
                resize_cpu,
            )

        # 5. BACKGROUND SUBTRACTION & FILTERING
        self.raw_mask = self.apply_background_subtraction_gpu(
            self.recycled_resize_mat,
            include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
            method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
            stream=self.bgs_stream,
        )

        if overall_frame_num <= debug_frame_limit:
            self.bgs_stream.waitForCompletion()
            cv2.imwrite(
                str(
                    stage_debug_dir
                    / f"frame_{overall_frame_num:04d}_stage5_history_mask.jpg"
                ),
                self.raw_mask.download(),
            )

        self._cuda_gaussian_filter.apply(
            self.raw_mask, self.d_blurred, stream=self.bgs_stream
        )

        if overall_frame_num <= debug_frame_limit:
            self.bgs_stream.waitForCompletion()
            cv2.imwrite(
                str(
                    stage_debug_dir / f"frame_{overall_frame_num:04d}_stage5b_blur.jpg"
                ),
                self.d_blurred.download(),
            )

        cv2.cuda.threshold(
            self.d_blurred,
            50,
            self.config.THRESHOLD_MAX_VALUE,
            cv2.THRESH_BINARY,
            self.thresh_mask,
            stream=self.bgs_stream,
        )

        if overall_frame_num <= debug_frame_limit:
            self.bgs_stream.waitForCompletion()
            cv2.imwrite(
                str(
                    stage_debug_dir
                    / f"frame_{overall_frame_num:04d}_stage6_threshold.jpg"
                ),
                self.thresh_mask.download(),
            )

        self.dilate_filter.apply(self.thresh_mask, self.clean_mask, self.bgs_stream)

        # [STAGE 7 DEBUG] Check Final Dilated Output Mask
        if overall_frame_num <= debug_frame_limit:
            self.bgs_stream.waitForCompletion()
            cv2.imwrite(
                str(
                    stage_debug_dir
                    / f"frame_{overall_frame_num:04d}_stage8_final_dilatemask.jpg"
                ),
                self.clean_mask.download(),
            )

        # inf_data = {
        #     "mask": self.clean_mask,
        #     "full_frame": device_frame,
        #     "frameNum": overall_frame_num,
        # }
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # 6. REGION OF INTEREST ANALYSIS
        limit_8K = 1280
        limit_640 = limit_8K / self.scale_x
        # start.record()
        raw_boxes = self.get_gpu_rois_by_area(
            overall_frame_num, self.clean_mask, max_candidates=50, limit_640=limit_640
        )
        # end.record()
        # torch.cuda.synchronize()
        # gpu_rois_by_area_ms = start.elapsed_time(end)

        if raw_boxes.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        raw_boxes = self.gpu_roi_padding(raw_boxes, method="uniform")
        # start.record()
        clean_640p = merge_boxes_gpu(
            raw_boxes,
            gap_limit=self.dist_thresh_640,
            size_limit=limit_640,
            max_cached_elements=self.max_cached_elements,
        )
        # end.record()
        # torch.cuda.synchronize()
        # merge_ms = start.elapsed_time(end)

        # main_app_logger.info(f"gpu_rois_by_area: {gpu_rois_by_area_ms:.3f} ms")
        # main_app_logger.info(f"merge_boxes_gpu:       {merge_ms:.3f} ms")

        if clean_640p.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        clean_640p[:, 0].clamp_(min=0, max=self.resize_w - 1)
        clean_640p[:, 1].clamp_(min=0, max=self.resize_h - 1)
        clean_640p[:, 2].clamp_(min=0, max=self.resize_w - 1)
        clean_640p[:, 3].clamp_(min=0, max=self.resize_h - 1)

        # OPTIMIZATION: IN-PLACE RESCALING
        # Avoid slicing arrays out sequentially into individual variables (xmin, ymin...)
        # Construct and scale standard boxes directly to prevent memory allocation chattering.
        standard_boxes = torch.stack(
            [clean_640p[:, 0], clean_640p[:, 1], clean_640p[:, 2], clean_640p[:, 3]],
            dim=1,
        )

        # if standard_boxes is None or standard_boxes.shape[0] == 0:
        #     return torch.empty((0, 4), device=self.device_input, dtype=torch.float32)

        # Multiply inplace and avoid manual `del` garbage sweeps to maintain maximum throughput
        bbs_full_res = (standard_boxes * self.scales_tensor).detach()
        return bbs_full_res

    # def get_sf_gpu_rois_v2(
    #     self, device_frame, overall_frame_num, max_candidates=100, limit_640=1280
    # ):
    #     debug_frame_limit = self.debug_frame_limit

    #     if overall_frame_num <= debug_frame_limit:
    #         stage_debug_dir = (
    #             self.result_dir / "debug_stages" / self._testMethodName / "roi_stages"
    #         )
    #         stage_debug_dir.mkdir(parents=True, exist_ok=True)

    #     # 2. BRIDGE THE PYTORCH TO OPENCV VRAM GAP
    #     if torch.is_tensor(device_frame):
    #         h_raw, w_raw, ch = device_frame.shape
    #         cuda_mem_ptr = device_frame.data_ptr()
    #         cv_type = cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1
    #         row_step_bytes = device_frame.stride()[0] * device_frame.element_size()
    #         src_gpu_mat = cv2.cuda.createGpuMatFromCudaMemory(
    #             h_raw, w_raw, cv_type, cuda_mem_ptr, step=row_step_bytes
    #         )
    #     else:
    #         src_gpu_mat = device_frame
    #         ch = src_gpu_mat.channels()

    #     if overall_frame_num <= debug_frame_limit:
    #         src_cpu = src_gpu_mat.download()
    #         if ch == 3:
    #             src_cpu = cv2.cvtColor(src_cpu, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(
    #             str(stage_debug_dir / f"frame_{overall_frame_num:04d}_stage1_src.jpg"),
    #             src_cpu,
    #         )
    #         del src_cpu

    #     # 3. STRIDED LAYOUT INITIALIZATION
    #     # if not hasattr(self, "raw_mask"):
    #     #     self.recycled_resize_mat = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1)
    #     #     self.raw_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
    #     #     self.thresh_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
    #     #     self.clean_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
    #     #     self.d_blurred = cv2.cuda.GpuMat(self.clean_mask.size(), cv2.CV_8UC1)

    #     # 4. RUN ASYNCHRONOUS DOWN-SAMPLING GATE
    #     cv2.cuda.resize(
    #         src_gpu_mat,
    #         dst=self.recycled_resize_mat,
    #         dsize=(self.resize_w, self.resize_h),
    #         interpolation=cv2.INTER_NEAREST,
    #         stream=self.bgs_stream,
    #     )
    #     isolated_resize_mat = self.recycled_resize_mat.clone()

    #     if overall_frame_num <= debug_frame_limit:
    #         self.bgs_stream.waitForCompletion()
    #         resize_cpu = self.recycled_resize_mat.download()
    #         if ch == 3:
    #             resize_cpu = cv2.cvtColor(resize_cpu, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(
    #             str(
    #                 stage_debug_dir / f"frame_{overall_frame_num:04d}_stage2_resize.jpg"
    #             ),
    #             resize_cpu,
    #         )

    #     # 5. BACKGROUND SUBTRACTION & FILTERING
    #     self.raw_mask = self.apply_background_subtraction_gpu(
    #         isolated_resize_mat,  # self.recycled_resize_mat,
    #         include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
    #         method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
    #         stream=self.bgs_stream,
    #     )

    #     if overall_frame_num <= debug_frame_limit:
    #         self.bgs_stream.waitForCompletion()
    #         cv2.imwrite(
    #             str(
    #                 stage_debug_dir
    #                 / f"frame_{overall_frame_num:04d}_stage5_history_mask.jpg"
    #             ),
    #             self.raw_mask.download(),
    #         )

    #     # self._cuda_gaussian_filter.apply(
    #     #     self.raw_mask, self.d_blurred, stream=self.bgs_stream
    #     # )
    #     # cv2.cuda.threshold(
    #     #     self.d_blurred,
    #     #     50,
    #     #     self.config.THRESHOLD_MAX_VALUE,
    #     #     cv2.THRESH_BINARY,
    #     #     self.thresh_mask,
    #     #     stream=self.bgs_stream,
    #     # )

    #     # self.dilate_filter.apply(self.thresh_mask, self.clean_mask, self.bgs_stream)
    #     # --- START: FUSED KERNEL EXECUTION ---
    #     # We bridge raw_mask and clean_mask to CuPy arrays (zero-copy)
    #     height, width = self.raw_mask.size()

    #     raw_mask_cp = cupy.ndarray(
    #         shape=(height, width),
    #         dtype=cupy.uint8,
    #         memptr=cupy.cuda.MemoryPointer(
    #             cupy.cuda.UnownedMemory(
    #                 self.raw_mask.cudaPtr(), self.raw_mask.step * height, self.raw_mask
    #             ),
    #             0,
    #         ),
    #         strides=(self.raw_mask.step, 1),
    #     )

    #     clean_mask_cp = cupy.ndarray(
    #         shape=(height, width),
    #         dtype=cupy.uint8,
    #         memptr=cupy.cuda.MemoryPointer(
    #             cupy.cuda.UnownedMemory(
    #                 self.clean_mask.cudaPtr(),
    #                 self.clean_mask.step * height,
    #                 self.clean_mask,
    #             ),
    #             0,
    #         ),
    #         strides=(self.clean_mask.step, 1),
    #     )

    #     # 3. Launch the Fused CUDA Kernel!
    #     # Grid/Block sizes optimized for RTX A6000 architecture
    #     block_dim = 16
    #     grid_size = (
    #         (width + block_dim - 1) // block_dim,
    #         (height + block_dim - 1) // block_dim,
    #     )
    #     block_size = (block_dim, block_dim)

    #     # --- START: FUSED KERNEL EXECUTION ---
    #     # Get the underlying CUDA stream pointer from the OpenCV stream
    #     # self.cupy_bgs_stream = cupy.cuda.ExternalStream(self.bgs_stream.cudaPtr())
    #     # self.fused_blur_thresh_kernel(
    #     #     grid_size,
    #     #     block_size,
    #     #     (
    #     #         self.raw_mask.cudaPtr(),      # const unsigned char* src
    #     #         self.thresh_mask.cudaPtr(),   # unsigned char* dst
    #     #         width,                        # int width
    #     #         height,                       # int height
    #     #         self.raw_mask.step,           # int src_step
    #     #         self.thresh_mask.step         # int dst_step
    #     #     ),
    #     #     stream=self.cupy_bgs_stream
    #     # )
    #     # self.dilate_filter.apply(self.thresh_mask, self.clean_mask, self.bgs_stream)

    #     # --- END: FUSED KERNEL EXECUTION ---

    #     # --- START: 3-in-1 FUSED KERNEL EXECUTION ---

    #     # 2. Wait for OpenCV BGS stream to finish before CuPy takes over
    #     # self.bgs_stream.waitForCompletion()
    #     self.cupy_bgs_stream = cupy.cuda.ExternalStream(self.bgs_stream.cudaPtr())

    #     # 4. Launch the single fused kernel on CuPy's default stream
    #     self.fused_3_in_1_kernel(
    #         grid_size,
    #         block_size,
    #         (
    #             raw_mask_cp,  # src
    #             clean_mask_cp,  # dst
    #             width,
    #             height,
    #             self.raw_mask.step,
    #             self.clean_mask.step,
    #         ),
    #         stream=self.cupy_bgs_stream,
    #     )

    #     # 5. Ensure CuPy kernel is finished before proceeding
    #     # import cupy as cp
    #     # cupy.cuda.Stream.null.synchronize()
    #     # --- END: FUSED KERNEL EXECUTION ---

    #     # [STAGE 7 DEBUG] Check Final Dilated Output Mask
    #     # if overall_frame_num <= debug_frame_limit:
    #     #     self.bgs_stream.waitForCompletion()
    #     #     cv2.imwrite(
    #     #         str(
    #     #             stage_debug_dir
    #     #             / f"frame_{overall_frame_num:04d}_stage8_final_dilatemask.jpg"
    #     #         ),
    #     #         self.clean_mask.download(),
    #     #     )

    #     # inf_data = {
    #     #     "mask": self.clean_mask,
    #     #     "full_frame": device_frame,
    #     #     "frameNum": overall_frame_num,
    #     # }
    #     # start = torch.cuda.Event(enable_timing=True)
    #     # end = torch.cuda.Event(enable_timing=True)

    #     # 6. REGION OF INTEREST ANALYSIS
    #     limit_8K = 1280
    #     limit_640 = limit_8K / self.scale_x
    #     # start.record()
    #     raw_boxes = self.get_gpu_rois_by_area(
    #         overall_frame_num, self.clean_mask, max_candidates=50, limit_640=limit_640
    #     )
    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # gpu_rois_by_area_ms = start.elapsed_time(end)

    #     if raw_boxes.shape[0] < 1:
    #         return torch.empty((0, 4), device=self.device_input)

    #     raw_boxes = self.gpu_roi_padding(raw_boxes, method="uniform")
    #     # start.record()
    #     clean_640p = merge_boxes_gpu(
    #         raw_boxes,
    #         gap_limit=self.dist_thresh_640,
    #         size_limit=limit_640,
    #         max_cached_elements=self.max_cached_elements,
    #     )
    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # merge_ms = start.elapsed_time(end)

    #     # main_app_logger.info(f"gpu_rois_by_area: {gpu_rois_by_area_ms:.3f} ms")
    #     # main_app_logger.info(f"merge_boxes_gpu:       {merge_ms:.3f} ms")

    #     if clean_640p.shape[0] < 1:
    #         return torch.empty((0, 4), device=self.device_input)

    #     clean_640p[:, 0].clamp_(min=0, max=self.resize_w - 1)
    #     clean_640p[:, 1].clamp_(min=0, max=self.resize_h - 1)
    #     clean_640p[:, 2].clamp_(min=0, max=self.resize_w - 1)
    #     clean_640p[:, 3].clamp_(min=0, max=self.resize_h - 1)

    #     # OPTIMIZATION: IN-PLACE RESCALING
    #     # Avoid slicing arrays out sequentially into individual variables (xmin, ymin...)
    #     # Construct and scale standard boxes directly to prevent memory allocation chattering.
    #     standard_boxes = torch.stack(
    #         [clean_640p[:, 0], clean_640p[:, 1], clean_640p[:, 2], clean_640p[:, 3]],
    #         dim=1,
    #     )

    #     # if standard_boxes is None or standard_boxes.shape[0] == 0:
    #     #     return torch.empty((0, 4), device=self.device_input, dtype=torch.float32)

    #     # Multiply inplace and avoid manual `del` garbage sweeps to maintain maximum throughput
    #     bbs_full_res = (standard_boxes * self.scales_tensor).detach()
    #     return bbs_full_res

    #  v3: reduce memory footprint
    def get_sf_gpu_rois(
        self, device_frame, overall_frame_num, max_candidates=100, limit_640=1280
    ):
        cupy.get_default_memory_pool().free_all_blocks()
        height, width = self.raw_mask.size()

        raw_mask_cp = cupy.ndarray(
            shape=(height, width),
            dtype=cupy.uint8,
            memptr=cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(
                    self.raw_mask.cudaPtr(),
                    self.raw_mask.step * height,
                    self.raw_mask,
                ),
                0,
            ),
            strides=(self.raw_mask.step, 1),
        )

        clean_mask_cp = cupy.ndarray(
            shape=(height, width),
            dtype=cupy.uint8,
            memptr=cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(
                    self.clean_mask.cudaPtr(),
                    self.clean_mask.step * height,
                    self.clean_mask,
                ),
                0,
            ),
            strides=(self.clean_mask.step, 1),
        )
        try:
            debug_frame_limit = self.debug_frame_limit

            if overall_frame_num <= debug_frame_limit:
                stage_debug_dir = (
                    self.result_dir
                    / "debug_stages"
                    / self._testMethodName
                    / "roi_stages"
                )
                stage_debug_dir.mkdir(parents=True, exist_ok=True)

            # 2. BRIDGE THE PYTORCH TO OPENCV VRAM GAP
            if torch.is_tensor(device_frame):
                h_raw, w_raw, ch = device_frame.shape
                cuda_mem_ptr = device_frame.data_ptr()
                cv_type = cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1
                row_step_bytes = device_frame.stride()[0] * device_frame.element_size()
                src_gpu_mat = cv2.cuda.createGpuMatFromCudaMemory(
                    h_raw, w_raw, cv_type, cuda_mem_ptr, step=row_step_bytes
                )
            else:
                src_gpu_mat = device_frame
                ch = src_gpu_mat.channels()

            if overall_frame_num <= debug_frame_limit:
                src_cpu = src_gpu_mat.download()
                if ch == 3:
                    src_cpu = cv2.cvtColor(src_cpu, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(
                        stage_debug_dir
                        / f"frame_{overall_frame_num:04d}_stage1_src.jpg"
                    ),
                    src_cpu,
                )
                del src_cpu

            # 1. Swap active buffers (Double Buffering)
            # This gives us a 100% thread-safe copy without allocating a single byte of new VRAM!
            # if self.use_buffer_A:
            #     active_resize_mat = self.recycled_resize_mat_A
            #     self.use_buffer_A = False
            # else:
            #     active_resize_mat = self.recycled_resize_mat_B
            #     self.use_buffer_A = True

            # 4. RUN ASYNCHRONOUS DOWN-SAMPLING GATE
            cv2.cuda.resize(
                src_gpu_mat,
                dst=self.recycled_resize_mat,  # active_resize_mat,
                dsize=(self.resize_w, self.resize_h),
                interpolation=cv2.INTER_NEAREST,
                stream=self.bgs_stream,
            )
            # isolated_resize_mat = self.recycled_resize_mat.clone()

            if overall_frame_num <= debug_frame_limit:
                self.bgs_stream.waitForCompletion()
                resize_cpu = self.recycled_resize_mat.download()
                if ch == 3:
                    resize_cpu = cv2.cvtColor(resize_cpu, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(
                        stage_debug_dir
                        / f"frame_{overall_frame_num:04d}_stage2_resize.jpg"
                    ),
                    resize_cpu,
                )
                del resize_cpu

            # 5. BACKGROUND SUBTRACTION & FILTERING
            self.raw_mask = self.apply_background_subtraction_gpu(
                self.recycled_resize_mat,
                include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
                method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
                stream=self.bgs_stream,
            )

            if overall_frame_num <= debug_frame_limit:
                self.bgs_stream.waitForCompletion()
                cv2.imwrite(
                    str(
                        stage_debug_dir
                        / f"frame_{overall_frame_num:04d}_stage5_history_mask.jpg"
                    ),
                    self.raw_mask.download(),
                )

            # --- START: FUSED KERNEL EXECUTION ---
            # We bridge raw_mask and clean_mask to CuPy arrays (zero-copy)
            # height, width = self.raw_mask.size()

            # raw_mask_cp = cupy.ndarray(
            #     shape=(height, width),
            #     dtype=cupy.uint8,
            #     memptr=cupy.cuda.MemoryPointer(
            #         cupy.cuda.UnownedMemory(
            #             self.raw_mask.cudaPtr(),
            #             self.raw_mask.step * height,
            #             self.raw_mask,
            #         ),
            #         0,
            #     ),
            #     strides=(self.raw_mask.step, 1),
            # )

            # clean_mask_cp = cupy.ndarray(
            #     shape=(height, width),
            #     dtype=cupy.uint8,
            #     memptr=cupy.cuda.MemoryPointer(
            #         cupy.cuda.UnownedMemory(
            #             self.clean_mask.cudaPtr(),
            #             self.clean_mask.step * height,
            #             self.clean_mask,
            #         ),
            #         0,
            #     ),
            #     strides=(self.clean_mask.step, 1),
            # )

            # 3. Launch the Fused CUDA Kernel!
            # Grid/Block sizes optimized for RTX A6000 architecture
            block_dim = 16
            grid_size = (
                (width + block_dim - 1) // block_dim,
                (height + block_dim - 1) // block_dim,
            )
            block_size = (block_dim, block_dim)

            # 2. Wait for OpenCV BGS stream to finish before CuPy takes over
            # self.bgs_stream.waitForCompletion()
            self.cupy_bgs_stream = cupy.cuda.ExternalStream(self.bgs_stream.cudaPtr())

            # 4. Launch the single fused kernel on CuPy's default stream
            self.fused_3_in_1_kernel(
                grid_size,
                block_size,
                (
                    raw_mask_cp,  # src
                    clean_mask_cp,  # dst
                    width,
                    height,
                    self.raw_mask.step,
                    self.clean_mask.step,
                ),
                stream=self.cupy_bgs_stream,
            )

            # 6. REGION OF INTEREST ANALYSIS
            limit_8K = 1280
            limit_640 = limit_8K / self.scale_x
            # start.record()
            raw_boxes = self.get_gpu_rois_by_area(
                overall_frame_num,
                self.clean_mask,
                max_candidates=50,
                limit_640=limit_640,
            )
            # end.record()
            # torch.cuda.synchronize()
            # gpu_rois_by_area_ms = start.elapsed_time(end)

            if raw_boxes.shape[0] < 1:
                return torch.empty((0, 4), device=self.device_input)

            raw_boxes = self.gpu_roi_padding(raw_boxes, method="uniform")
            # start.record()
            clean_640p = merge_boxes_gpu(
                raw_boxes,
                gap_limit=self.dist_thresh_640,
                size_limit=limit_640,
                max_cached_elements=self.max_cached_elements,
            )
            # end.record()
            # torch.cuda.synchronize()
            # merge_ms = start.elapsed_time(end)

            # main_app_logger.info(f"gpu_rois_by_area: {gpu_rois_by_area_ms:.3f} ms")
            # main_app_logger.info(f"merge_boxes_gpu:       {merge_ms:.3f} ms")

            if clean_640p.shape[0] < 1:
                return torch.empty((0, 4), device=self.device_input)

            clean_640p[:, 0].clamp_(min=0, max=self.resize_w - 1)
            clean_640p[:, 1].clamp_(min=0, max=self.resize_h - 1)
            clean_640p[:, 2].clamp_(min=0, max=self.resize_w - 1)
            clean_640p[:, 3].clamp_(min=0, max=self.resize_h - 1)

            # OPTIMIZATION: IN-PLACE RESCALING
            # Avoid slicing arrays out sequentially into individual variables (xmin, ymin...)
            # Construct and scale standard boxes directly to prevent memory allocation chattering.
            # standard_boxes = torch.stack(
            #     [
            #         clean_640p[:, 0],
            #         clean_640p[:, 1],
            #         clean_640p[:, 2],
            #         clean_640p[:, 3],
            #     ],
            #     dim=1,
            # )
            bbs_full_res = (
                torch.stack(
                    [
                        clean_640p[:, 0],
                        clean_640p[:, 1],
                        clean_640p[:, 2],
                        clean_640p[:, 3],
                    ],
                    dim=1,
                )
                * self.scales_tensor
            ).detach()

            # if standard_boxes is None or standard_boxes.shape[0] == 0:
            #     return torch.empty((0, 4), device=self.device_input, dtype=torch.float32)

            # Multiply inplace and avoid manual `del` garbage sweeps to maintain maximum throughput
            # bbs_full_res = (standard_boxes * self.scales_tensor).detach()
        finally:
            # 2. Release temporary OpenCV GpuMats. Using 'in locals()' is safer.
            if "raw_mask_cp" in locals():
                del raw_mask_cp
            if "clean_mask_cp" in locals():
                del clean_mask_cp
            if "src_gpu_mat" in locals():
                del src_gpu_mat
            # if "isolated_resize_mat" in locals():
            #     del isolated_resize_mat

            del device_frame

            if "raw_boxes" in locals():
                del raw_boxes
            if "clean_640p" in locals():
                del clean_640p

            # 3. Force CuPy to release all unused memory back to the system.
            # This is the most critical step for preventing inter-library OOM errors.
            cupy.get_default_memory_pool().free_all_blocks()
        return bbs_full_res

    def run_gpu_pipeline(
        self, device_frame, overall_frame_num, frame_in_clip_count=0, gt_boxes=None
    ):
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}
        metadata = {}

        # Get ROIs
        if self.timer_enabled:
            self.sf_start.record(self.inference_stream)

        bgs_input_frame = (
            device_frame.byte() if torch.is_tensor(device_frame) else device_frame
        )
        bbs_full_res = self.get_sf_gpu_rois(bgs_input_frame, overall_frame_num)

        if self.timer_enabled:
            self.sf_end.record(self.inference_stream)

        # metrics["bbs"] = bbs_full_res
        # Detach the matrix coordinates cleanly out of the tracking graph immediately
        if bbs_full_res is not None:
            if torch.is_tensor(bbs_full_res):
                metrics["bbs"] = bbs_full_res.detach().cpu().numpy()
            elif isinstance(bbs_full_res, np.ndarray):
                metrics["bbs"] = bbs_full_res
        else:
            metrics["bbs"] = np.empty((0, 4), dtype=np.float32)

        merged, det_frame = self.format_bbs_and_frame_4_detection(
            bbs_full_res, device_frame
        )
        motion_detected = len(merged) > 0 if merged is not None else False
        metrics["batch_density"] = len(merged) if merged is not None else 0

        if (
            self.config.DEBUG_FLAG
            and overall_frame_num <= self.config.DEBUG_FRAME_LIMIT
        ):
            self.debug_save_mask(
                det_frame, overall_frame_num, rois=merged, gt_boxes=gt_boxes
            )

        if not self.config.DISABLE_DETECTION:
            # --- 3. MODEL INFERENCE TIMING BLOCK ---
            if self.device_input == "cuda" and self.timer_enabled:
                self.det_start.record(self.inference_stream)
            # elif self.timer_enabled:
            #     t_start = time.perf_counter()

            # Get Detection Metadata
            if self.config.DETECTION_TYPE != "motion":  # Object detection
                metadata, _ = self.get_detections(
                    det_frame,
                    frame_in_clip_count,
                    merged=merged,
                    thickness=self.config.THICKNESS,
                    device_input=self.config.device_input,
                )
                # num_objs = len(metadata.keys())
            else:  # Motion: Smart filtering results
                # 8k bb to 640
                metadata = self.motion2metadata(merged, frame_in_clip_count)

            if self.device_input == "cuda" and self.timer_enabled:
                self.det_end.record(self.inference_stream)
                # self.inference_stream.synchronize()

                # Lock-free check: wait for event completion without blocking the CPU GIL
                while not self.det_end.query():
                    # Yield GIL to let file-writers and thread-pools work
                    time.sleep(0.001)
                # self.det_end.synchronize()

                # Full-frame YOLO baseline tracks the elapsed time from t_start on page 20, line 1273
                metrics["sf_time"] = self.sf_start.elapsed_time(self.sf_end)
                metrics["det_time"] = self.det_start.elapsed_time(self.det_end)
        elif self.device_input == "cuda" and self.timer_enabled:
            metrics["sf_time"] = self.sf_start.elapsed_time(self.sf_end)

            # elif self.timer_enabled:
            #     # CPU Path Execution: Must use standard wall-clock timing loops to avoid CUDA Event errors
            #     metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0

        del merged, bbs_full_res
        return metrics, metadata, det_frame, motion_detected

    # v1
    def find_contours_gpu_equivalent(self, mask_gpu_mat, stream=None, limit_640=None):
        """
        Refactored allocation-free GPU bounding box contour extraction tree.
        Bypasses dynamic VRAM creation paths entirely.
        """

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        w, h = mask_gpu_mat.size()
        ptr = mask_gpu_mat.cudaPtr()
        pitch_bytes = mask_gpu_mat.step

        mask_cp = cupy.ndarray(
            (h, w),
            dtype=cupy.uint8,
            memptr=cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(ptr, pitch_bytes * h, mask_gpu_mat), 0
            ),
            strides=(pitch_bytes, 1),
        )

        stream_ptr = stream.cudaPtr() if stream else 0

        with cupy.cuda.ExternalStream(stream_ptr):
            if (
                not hasattr(self, "_labeled_scratch")
                or self._labeled_scratch.shape != (h, w)
                or not self._labeled_scratch.flags.c_contiguous
            ):
                # Explicitly force order='C' (Row-Major Linear Stride alignment) at initialization.
                # This completely eliminates internal syncdetect regular expression matching stutters!
                self._labeled_scratch = cupy.empty((h, w), dtype=cupy.int32, order="C")

            # start = cupy.cuda.Event()
            # end = cupy.cuda.Event()
            # start.record()
            # 2. FAST IN-PLACE LABELING
            # Because output is guaranteed to be a linear memory layout, SciPy processes it with zero device syncs
            # TODO: Custom CUDA kernel to replace b/c label algorithm is costly
            num_labels = cupyx.scipy.ndimage.label(
                mask_cp, structure=self.cupy_structure, output=self._labeled_scratch
            )
            # end.record()
            # end.synchronize()
            # ms = cupy.cuda.get_elapsed_time(start, end)
            # main_app_logger.info(f"cupyx label: {ms:.3f} ms")

            if num_labels == 0:
                return torch.empty((0, 4), device="cuda")

            # Prevent array overflows if features spike abnormally
            safe_labels = min(num_labels + 1, self._max_labels)

            # Quick, specialized in-place fills over our persistent pools (0 loop churn)
            self._x1_pool[:safe_labels].fill(w)
            self._y1_pool[:safe_labels].fill(h)
            self._x2_pool[:safe_labels].fill(-1)
            self._y2_pool[:safe_labels].fill(-1)

            # self._x1_pool[:safe_labels] = w
            # self._y1_pool[:safe_labels] = h
            # self._x2_pool[:safe_labels] = -1
            # self._y2_pool[:safe_labels] = -1

            # Extract references for your custom kernel execution map
            pitch_elements = self._labeled_scratch.strides[0] // 4

            tpb = (16, 16, 1)
            bpg = ((w + tpb[0] - 1) // tpb[0], (h + tpb[1] - 1) // tpb[1], 1)

            # 3. DIRECT DRIVER FUNCTION CALL (Maintains your accurate get_bounds logic)
            self._raw_bounds_function(
                bpg,
                tpb,
                (
                    self._labeled_scratch.data.ptr,  # const int* labeled
                    pitch_elements,  # int pitch
                    w,  # int w
                    h,  # int h
                    num_labels,  # int num_labels
                    self._x1_pool[:safe_labels].data.ptr,  # int* x1
                    self._y1_pool[:safe_labels].data.ptr,  # int* y1
                    self._x2_pool[:safe_labels].data.ptr,  # int* x2
                    self._y2_pool[:safe_labels].data.ptr,  # int* y2
                ),
            )

            # 4. ELIMINATE COLUMN_STACK ALLOCATION OVERHEAD (Replaces lines 91-105)
            # Your previous code allocated memory on the heap every frame via cupy.column_stack.
            # Instead, map the pre-allocated pool buffers directly into contiguous PyTorch tensor views!
            x1_t = torch.as_tensor(self._x1_pool[1:safe_labels], device="cuda")
            y1_t = torch.as_tensor(self._y1_pool[1:safe_labels], device="cuda")
            x2_t = torch.as_tensor(self._x2_pool[1:safe_labels], device="cuda")
            y2_t = torch.as_tensor(self._y2_pool[1:safe_labels], device="cuda")

            # Stack tensors natively using unified memory views without copies or host syncs
            return torch.stack((x1_t, y1_t, x2_t, y2_t), dim=1).float()

    # def find_contours_gpu_equivalent_v3(self, clean_mask, stream=None, limit_640=1280):
    #     """
    #     Extremely fast, zero-allocation GPU boundary extraction.
    #     Bypasses OpenCV's costly contour allocations entirely.
    #     """
    #     # Convert the OpenCV GpuMat directly to a PyTorch GPU Tensor view (Zero-Copy)
    #     # This does not allocate any new VRAM!
    #     # mask_tensor = torch.as_tensor(
    #     #     clean_mask,
    #     #     device="cuda"
    #     # )
    #     height, width = clean_mask.size()

    #     # 1. Grab the raw C++ VRAM pointer from the OpenCV GpuMat
    #     # and wrap it in a CuPy array without copying the data
    #     mask_cp = cupy.ndarray(
    #         shape=(height, width),
    #         dtype=cupy.uint8,
    #         memptr=cupy.cuda.MemoryPointer(
    #             cupy.cuda.UnownedMemory(
    #                 clean_mask.cudaPtr(),
    #                 clean_mask.step * height,
    #                 clean_mask
    #             ),
    #             0
    #         ),
    #         strides=(clean_mask.step, 1)
    #     )
    #     mask_tensor = from_dlpack(mask_cp.toDlpack())

    #     # 1. Extract the coordinates of all active motion pixels (Non-zero coordinates)
    #     # We write directly into our pre-allocated coordinates scratchpad using a view
    #     nz = torch.nonzero(mask_tensor, out=self.coords_scratchpad)

    #     if nz.shape[0] == 0:
    #         return torch.empty((0, 4), dtype=torch.float32, device="cuda")

    #     # 2. Extract boundaries using vectorized Min/Max operations
    #     # This completely avoids creating temporary contour objects!
    #     y_coords = nz[:, 0]
    #     x_coords = nz[:, 1]

    #     # Find the global min/max coordinates of the motion region
    #     x1 = x_coords.min()
    #     y1 = y_coords.min()
    #     x2 = x_coords.max()
    #     y2 = y_coords.max()

    #     # 3. Write directly to our pre-allocated static output box
    #     self.static_boxes_out[0, 0] = x1
    #     self.static_boxes_out[0, 1] = y1
    #     self.static_boxes_out[0, 2] = x2
    #     self.static_boxes_out[0, 3] = y2

    #     # Return a view of the active boxes (Zero new allocations)
    #     return self.static_boxes_out[:1]

    # def find_contours_gpu_equivalent_v2(self, mask_gpu_mat, stream=None, limit_640=None):
    #     """
    #     Replaces the CuPy version with a 100% PyTorch-native implementation.
    #     This completely eliminates CuPy-PyTorch context-switching overhead
    #     and utilizes parallel GPU reduction via scatter_reduce.
    #     """
    #     # We manually construct the __cuda_array_interface__ to share GPU memory pointers
    #     # without any host-device copies.

    #     # 1. Gather GpuMat metadata
    #     height, width = mask_gpu_mat.size()
    #     gpu_ptr = mask_gpu_mat.cudaPtr()
    #     step = mask_gpu_mat.step  # Number of bytes per row (stride)

    #     # Instantly wrap the pointer into a PyTorch GPU Tensor (0 ms latency!)
    #     holder = GPUHolder({
    #         'shape': (height, width),
    #         'typestr': '|u1',  # OpenCV's CV_8UC1 is equivalent to uint8
    #         'data': (gpu_ptr, False), # Pointer address, Read-only=False
    #         'strides': (step, 1),
    #         'version': 3
    #     })
    #     mask_tensor = torch.as_tensor(holder, device=self.device_input)
    #     # =========================================================================

    #     # 2. Convert mask to boolean for segmentation
    #     binary_mask = mask_tensor > 0

    #     # 3. Use OpenCV's highly optimized CPU CC labeling (faster than CuPy for small labels/warmups)
    #     # Or if OpenCV's CUDA CC labeling is available, dispatch there.
    #     # This gives a fast, reliable label grid.
    #     num_labels, labels = cv2.connectedComponents(
    #         binary_mask.cpu().numpy().astype(np.uint8),
    #         connectivity=8
    #     )

    #     # If there are only background pixels, return an empty coordinate tensor
    #     if num_labels <= 1:
    #         return torch.empty((0, 4), device=self.device_input, dtype=torch.float32)

    #     # 4. Push label grid back to the active GPU
    #     labels_gpu = torch.as_tensor(labels, device=self.device_input)

    #     # 5. Pre-allocate coordinate bounds tensors on the GPU
    #     # We use standard 32-bit integers to keep memory operations light
    #     max_val = torch.iinfo(torch.int32).max
    #     x1 = torch.full((num_labels,), max_val, device=self.device_input, dtype=torch.int32)
    #     y1 = torch.full((num_labels,), max_val, device=self.device_input, dtype=torch.int32)
    #     x2 = torch.full((num_labels,), -1, device=self.device_input, dtype=torch.int32)
    #     y2 = torch.full((num_labels,), -1, device=self.device_input, dtype=torch.int32)

    #     # 6. Extract raw indices of all active foreground pixels on GPU
    #     # This keeps the coordinates completely in memory
    #     y_coords, x_coords = torch.nonzero(labels_gpu, as_tuple=True)
    #     label_vals = labels_gpu[y_coords, x_coords]

    #     # 7. Perform high-speed parallel GPU reductions to find min/max coordinates
    #     # 'amin' and 'amax' are computed concurrently across all threads
    #     x1.scatter_reduce_(0, label_vals, x_coords.int(), reduce="amin", include_self=False)
    #     y1.scatter_reduce_(0, label_vals, y_coords.int(), reduce="amin", include_self=False)
    #     x2.scatter_reduce_(0, label_vals, x_coords.int(), reduce="amax", include_self=False)
    #     y2.scatter_reduce_(0, label_vals, y_coords.int(), reduce="amax", include_self=False)

    #     # 8. Stack bounds and discard index 0 (which represents the background)
    #     # Returns shape: (N, 4) -> [x1, y1, x2, y2] matching your downstream merge_boxes format
    #     boxes = torch.stack((x1[1:], y1[1:], x2[1:], y2[1:]), dim=1).float()

    #     return boxes

    # def find_contours_gpu_equivalent_v3(self, mask_gpu_mat, stream=None, limit_640=None):
    #     """
    #     100% GPU-native connected components labeling.
    #     Combines CuPy labeling with PyTorch-native vectorized bounding box reduction.
    #     """
    #     # import cupy
    #     # import cupyx.scipy.ndimage
    #     # import torch

    #     # 1. Zero-copy bridge from OpenCV GpuMat to CuPy array on the GPU
    #     height, width = mask_gpu_mat.size()

    #     mask_cp = cupy.ndarray(
    #         shape=(height, width),
    #         dtype=cupy.uint8,
    #         memptr=cupy.cuda.MemoryPointer(
    #             cupy.cuda.UnownedMemory(mask_gpu_mat.cudaPtr(), mask_gpu_mat.step * height, mask_gpu_mat), 0
    #         ),
    #         strides=(mask_gpu_mat.step, 1)
    #     )

    #     # 2. Perform high-speed connected components labeling on GPU using CuPy
    #     structure = cupy.ones((3, 3), dtype=cupy.int32)
    #     labeled_mask, num_labels = cupyx.scipy.ndimage.label(mask_cp > 0, structure=structure)

    #     if num_labels == 0:
    #         return torch.empty((0, 4), device=self.device_input, dtype=torch.float32)

    #     # =========================================================================
    #     # 3. Zero-copy bridge from labeled CuPy array to PyTorch GPU Tensor
    #     # =========================================================================
    #     # Since labeled_mask is a CuPy array, we can access its __cuda_array_interface__
    #     # to expose the memory block directly to PyTorch without any copies!
    #     labels_gpu = torch.as_tensor(labeled_mask, device=self.device_input)

    #     # 4. Pre-allocate coordinate bounds tensors on the GPU
    #     max_val = torch.iinfo(torch.int32).max
    #     x1 = torch.full((num_labels + 1,), max_val, device=self.device_input, dtype=torch.int32)
    #     y1 = torch.full((num_labels + 1,), max_val, device=self.device_input, dtype=torch.int32)
    #     x2 = torch.full((num_labels + 1,), -1, device=self.device_input, dtype=torch.int32)
    #     y2 = torch.full((num_labels + 1,), -1, device=self.device_input, dtype=torch.int32)

    #     # 5. Extract raw indices of all active foreground pixels on the GPU
    #     y_coords, x_coords = torch.nonzero(labels_gpu, as_tuple=True)
    #     label_vals = labels_gpu[y_coords, x_coords].long() # Cast to long for indexing

    #     # 6. Parallel GPU reduction to find bounding coordinates for all labels at once
    #     x1.scatter_reduce_(0, label_vals, x_coords.int(), reduce="amin", include_self=False)
    #     y1.scatter_reduce_(0, label_vals, y_coords.int(), reduce="amin", include_self=False)
    #     x2.scatter_reduce_(0, label_vals, x_coords.int(), reduce="amax", include_self=False)
    #     y2.scatter_reduce_(0, label_vals, y_coords.int(), reduce="amax", include_self=False)

    #     # 7. Stack bounds, skipping index 0 (the background label)
    #     # Returns shape: (N, 4) -> [x1, y1, x2, y2]
    #     boxes = torch.stack((x1[1:], y1[1:], x2[1:], y2[1:]), dim=1).float()

    #     return boxes

    def get_gpu_rois_by_area(self, frameNum, mask, max_candidates=100, limit_640=1280):
        # Extract true spatial constraints straight from the active mask object footprint
        if torch.is_tensor(mask):
            mask_h, mask_w = mask.shape[-2:]
        elif isinstance(mask, cv2.cuda.GpuMat):
            # cv2.cuda.GpuMat.size() returns a tuple of (width, height) standard formatting
            mask_w, mask_h = mask.size()
        else:
            mask_h, mask_w = mask.shape[:2]

        # This prevents find_contours_gpu_equivalent from mutating the mask variables used by other threads.
        if isinstance(mask, cv2.cuda.GpuMat):
            # .clone() allocates a new C++ memory surface and forces full continuity
            isolated_kernel_mask = mask  # .clone()
        elif torch.is_tensor(mask):
            isolated_kernel_mask = mask  # .clone().contiguous()
        else:
            isolated_kernel_mask = mask  # .copy()

        # Get raw boxes from mask (Direct VRAM bridge)
        boxes_gpu = self.find_contours_gpu_equivalent(
            isolated_kernel_mask,
            stream=self.bgs_stream,
            limit_640=limit_640,  # 640*1.5,
        )

        # --- FIX: ELIMINATE STREAM RACE ---
        if boxes_gpu is None or len(boxes_gpu) == 0:
            return torch.empty((0, 4), device=self.device_input)

        # if len(boxes_gpu) > max_candidates:  # Adjust threshold based on target max objects
        #     # Prioritize or slice to prevent merge_boxes_gpu from thrashing scatter_reduce_
        #     boxes_gpu = boxes_gpu[:max_candidates]

        # Wrap existing GPU memory as a float tensor (Zero Copy)
        raw_boxes = torch.as_tensor(boxes_gpu, device=self.device_input).float()
        # if self.device_input == "cuda":
        #     # Wrap the native device handle and IMMEDIATELY append .clone()
        #     # This allocates a brand new, physically isolated VRAM block to secure the bounding boxes
        #     raw_boxes = (
        #         torch.as_tensor(boxes_gpu, device=self.device_input).float().clone()
        #     )
        # else:
        #     raw_boxes = torch.as_tensor(boxes_gpu, device=self.device_input).float()

        if raw_boxes is not None and len(raw_boxes) > 0:
            # Vectorized Pre-Filter (Removes noise blobs before merging)
            w = raw_boxes[:, 2] - raw_boxes[:, 0]
            h = raw_boxes[:, 3] - raw_boxes[:, 1]
            # mask_filter = (w > 2) & (h > 2) & (w * h > self.min_contour_area) & (w < mask_w) & (h < mask_h)
            mask_filter = (
                (w <= self.max_roi_w)
                & (h <= self.max_roi_h)
                & (w >= self.min_roi_w)
                & (h >= self.min_roi_h)
            )
            # Ensure both structures match dimensions along the indexing axis
            # if raw_boxes.shape[0] == mask_filter.shape[0] and raw_boxes.shape[0] > 0:
            #     raw_boxes = raw_boxes[mask_filter]
            # else:
            #     # Short-circuit directly to an empty tensor if shapes don't match or are 0
            #     raw_boxes = torch.empty((0, 4), device=self.device_input, dtype=torch.float32)
            if raw_boxes.shape[0] > 0:
                raw_boxes = raw_boxes[mask_filter]

            # Guard rails: Clamp boundaries in-place to stay safely within the 8K master frame
            # Indices 0 and 2 are X coordinates bounded by frame width; 1 and 3 are Y coordinates bounded by frame height
            raw_boxes[:, 0].clamp_(min=0, max=self.resize_w - 1)
            raw_boxes[:, 1].clamp_(min=0, max=self.resize_h - 1)
            raw_boxes[:, 2].clamp_(min=0, max=self.resize_w - 1)
            raw_boxes[:, 3].clamp_(min=0, max=self.resize_h - 1)

            # Re-assign back to your pipeline's tracking variable
            # raw_boxes = padded_tensor

        # Prevents N^2 distance matrix from exploding during high noise
        # if raw_boxes.shape[0] > max_candidates:
        #     # Prioritize the largest blobs (most likely to be drones)
        #     areas = (raw_boxes[:, 2] - raw_boxes[:, 0]) * (
        #         raw_boxes[:, 3] - raw_boxes[:, 1]
        #     )
        #     flat_areas = areas.view(-1)
        #     _, indices = torch.topk(
        #         flat_areas, k=min(max_candidates, flat_areas.shape[0]), dim=-1
        #     )
        #     # _, indices = torch.topk(areas, max_candidates)
        #     raw_boxes = raw_boxes[indices]
        num_boxes = raw_boxes.shape[0]
        if num_boxes > max_candidates:
            areas = (raw_boxes[:, 2] - raw_boxes[:, 0]) * (
                raw_boxes[:, 3] - raw_boxes[:, 1]
            )
            _, indices = torch.topk(areas, k=max_candidates, dim=0)
            raw_boxes = raw_boxes[indices]
        return raw_boxes

    def gpu_roi_padding(self, raw_boxes, method="uniform"):
        """
        raw_boxes: in resize space (640)
        """

        if method == "uniform":
            # pad using scale factor
            widths = raw_boxes[:, 2] - raw_boxes[:, 0]
            heights = raw_boxes[:, 3] - raw_boxes[:, 1]
            # pad_w = widths * (PADDING_SCALE / 2.0)
            # pad_h = heights * (PADDING_SCALE / 2.0)
            # Add a minimum 3px clamp to ensure tiny objects don't lose their margins
            pad_w = torch.clamp(widths * (PADDING_SCALE / 2.0), min=3.0)
            pad_h = torch.clamp(heights * (PADDING_SCALE / 2.0), min=3.0)
            padding_mask = torch.stack([-pad_w, -pad_h, pad_w, pad_h], dim=1)

            # Apply padding to all bounding boxes concurrently via broad-vector math
            raw_boxes = raw_boxes + padding_mask
        elif method == "modelsize":
            # extend box up to MODEL size
            # if box is larger, keep as-is
            scaley_8Kto640 = self.config.MODEL_H / self.frame_height
            scalex_8Kto640 = self.config.MODEL_W / self.frame_width
            model_w_640 = int(self.config.MODEL_W * scalex_8Kto640)
            model_h_640 = int(self.config.MODEL_H * scaley_8Kto640)

            widths = raw_boxes[:, 2] - raw_boxes[:, 0]
            heights = raw_boxes[:, 3] - raw_boxes[:, 1]
            # if (widths,heights) == (model_w_640,model_h_640) or widths > model_w_640 or heights > model_h_640:
            #     pass  # Keep as is
            # else:
            x_centers = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2.0
            y_centers = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2.0

            # Determine target dimensions (clamp to target size if smaller)
            new_widths = torch.clamp(widths, min=model_w_640)
            new_heights = torch.clamp(heights, min=model_h_640)

            # Calculate new coordinates from the centers
            new_x1 = x_centers - (new_widths / 2.0)
            new_y1 = y_centers - (new_heights / 2.0)
            new_x2 = x_centers + (new_widths / 2.0)
            new_y2 = y_centers + (new_heights / 2.0)

            # 4. Stack into a tensor matching raw_boxes shape
            raw_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
        elif method == "pixel":
            # pad using scale factor
            # widths = raw_boxes[:, 2] - raw_boxes[:, 0]
            # heights = raw_boxes[:, 3] - raw_boxes[:, 1]
            # # pad_w = widths * (PADDING_SCALE / 2.0)
            # # pad_h = heights * (PADDING_SCALE / 2.0)
            # Add a minimum 3px clamp to ensure tiny objects don't lose their margins
            # padding_mask = torch.stack([-PADDING_PX, -PADDING_PX, PADDING_PX, PADDING_PX], dim=1)

            # Apply padding to all bounding boxes concurrently via broad-vector math
            # raw_boxes = raw_boxes + padding_mask
            # padding = 5  # self.config.ROI_BB_FULL_RES_PADDING /  self.scale_x
            padding_scale = 0.01
            raw_boxes[:, 0] -= int(padding_scale * self.frame_width)
            raw_boxes[:, 1] -= int(padding_scale * self.frame_height)
            raw_boxes[:, 2] += int(padding_scale * self.frame_width)
            raw_boxes[:, 3] += int(padding_scale * self.frame_height)

        return raw_boxes

    def filter_contained_boxes(self, boxes, overlap_thresh=0.9, max_elements=2000):
        """
        True Zero-Copy Anchor Filter. Eliminates torch.stack stream synchronization
        stalls by utilizing a pre-allocated boolean tracking register map.
        Supports both CPU and GPU execution profiles natively.
        """
        if boxes.shape[0] <= 1:
            return boxes

        # 1. Extract dimensions safely using vectorized layout views
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        areas = w * h

        valid_mask = (
            (w < self.max_roi_w)
            & (h < self.max_roi_h)
            & (w >= self.min_roi_w)
            & (h >= self.min_roi_h)
        )
        boxes = boxes[valid_mask]
        areas = areas[valid_mask]

        N = boxes.shape[0]
        if N <= 1:
            return boxes

        # 2. Sort structural metrics into our pre-allocated scratch index register
        order = self._filter_order_scratch[:N]
        torch.argsort(areas, stable=False, dim=0, descending=True, out=order)

        # 🚀 REPLACEMENT: Use an in-place boolean tracking mask instead of a Python list
        # to collect survival flags without triggering heap fragmentation or synchronization locks.
        if (
            not hasattr(self, "_filter_survival_mask")
            or self._filter_survival_mask.shape[0] < max_elements
        ):
            self._filter_survival_mask = torch.zeros(
                (max_elements,), dtype=torch.bool, device=boxes.device
            )

        survival_mask = self._filter_survival_mask[:N].zero_()

        # 3. User-Space Evaluation Loop
        while order.numel() > 0:
            i = order[0]
            survival_mask[i] = (
                True  # Mark this anchor index as verified on-device instantly
            )

            if order.numel() == 1:
                break

            order_tail = order[1:]
            num_tail = order_tail.numel()

            # Enforce static VRAM outputs using target 'out=' parameter parameters
            torch.max(
                boxes[i, 0],
                boxes[order_tail, 0],
                out=self._filter_scratch_x1[:num_tail],
            )
            torch.max(
                boxes[i, 1],
                boxes[order_tail, 1],
                out=self._filter_scratch_y1[:num_tail],
            )
            torch.min(
                boxes[i, 2],
                boxes[order_tail, 2],
                out=self._filter_scratch_x2[:num_tail],
            )
            torch.min(
                boxes[i, 3],
                boxes[order_tail, 3],
                out=self._filter_scratch_y2[:num_tail],
            )

            inter_w = (
                self._filter_scratch_x2[:num_tail] - self._filter_scratch_x1[:num_tail]
            ).clamp_(min=0)
            inter_h = (
                self._filter_scratch_y2[:num_tail] - self._filter_scratch_y1[:num_tail]
            ).clamp_(min=0)
            inter_area = inter_w * inter_h

            torch.div(
                inter_area,
                (areas[order_tail] + 1e-6),
                out=self._filter_scratch_ioa[:num_tail],
            )

            mask = self._filter_scratch_keep[:num_tail]
            torch.le(self._filter_scratch_ioa[:num_tail], overlap_thresh, out=mask)

            # Advance tracking window slice cleanly to the filtered child nodes
            order = order_tail[mask]

        # 🚀 REPLACEMENT: Slice the original tensor layout with the boolean mask directly.
        # This keeps the execution pipeline loopless and avoids cross-hardware synchronization stalls.
        return boxes[survival_mask]

    def cpu_roi_padding(self, coords_xywh, method="uniform"):
        """
        raw_boxes: in resize space (640)
        """

        if method == "uniform":
            # pad using scale factor
            widths = coords_xywh[:, 2]
            heights = coords_xywh[:, 3]
            # pad_w = widths * (PADDING_SCALE / 2.0)
            # pad_h = heights * (PADDING_SCALE / 2.0)
            # Add a minimum 3px clamp to ensure tiny objects don't lose their margins
            pad_w = torch.clamp(widths * (PADDING_SCALE / 2.0), min=3.0)
            pad_h = torch.clamp(heights * (PADDING_SCALE / 2.0), min=3.0)
            # padding_mask = torch.stack([-pad_w, -pad_h, pad_w, pad_h], dim=1)

            x1 = coords_xywh[:, 0] - pad_w
            y1 = coords_xywh[:, 1] - pad_h
            x2 = (coords_xywh[:, 0] + widths) + pad_w
            y2 = (coords_xywh[:, 1] + heights) + pad_h
            raw_boxes = torch.stack([x1, y1, x2, y2], dim=1)

        elif method == "modelsize":
            # extend box up to MODEL size
            # if box is larger, keep as-is
            scaley_8Kto640 = self.config.MODEL_H / self.frame_height
            scalex_8Kto640 = self.config.MODEL_W / self.frame_width
            model_w_640 = int(self.config.MODEL_W * scalex_8Kto640)
            model_h_640 = int(self.config.MODEL_H * scaley_8Kto640)

            widths = coords_xywh[:, 2]
            heights = coords_xywh[:, 3]
            # if (widths,heights) == (model_w_640,model_h_640) or widths > model_w_640 or heights > model_h_640:
            #     pass  # Keep as is
            # else:
            x_centers = (coords_xywh[:, 0] + widths) / 2.0
            y_centers = (coords_xywh[:, 1] + heights) / 2.0

            # Determine target dimensions (clamp to target size if smaller)
            new_widths = torch.clamp(widths, min=model_w_640)
            new_heights = torch.clamp(heights, min=model_h_640)

            # Calculate new coordinates from the centers
            new_x1 = x_centers - (new_widths / 2.0)
            new_y1 = y_centers - (new_heights / 2.0)
            new_x2 = x_centers + (new_widths / 2.0)
            new_y2 = y_centers + (new_heights / 2.0)

            # 4. Stack into a tensor matching raw_boxes shape
            raw_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

        elif method == "pixel":
            # pad using scale factor
            # widths = raw_boxes[:, 2] - raw_boxes[:, 0]
            # heights = raw_boxes[:, 3] - raw_boxes[:, 1]
            # # pad_w = widths * (PADDING_SCALE / 2.0)
            # # pad_h = heights * (PADDING_SCALE / 2.0)
            # Add a minimum 3px clamp to ensure tiny objects don't lose their margins
            # padding_mask = torch.stack([-PADDING_PX, -PADDING_PX, PADDING_PX, PADDING_PX], dim=1)

            # Apply padding to all bounding boxes concurrently via broad-vector math
            # raw_boxes = raw_boxes + padding_mask
            # padding = 5  # self.config.ROI_BB_FULL_RES_PADDING /  self.scale_x
            padding_scale = 0.01
            widths = coords_xywh[:, 2]
            heights = coords_xywh[:, 3]
            pad_w = int(padding_scale * self.frame_width)
            pad_h = int(padding_scale * self.frame_height)
            # raw_boxes[:, 2] += int(padding_scale * self.frame_width)
            # raw_boxes[:, 3] += int(padding_scale * self.frame_height)

            x1 = coords_xywh[:, 0] - pad_w
            y1 = coords_xywh[:, 1] - pad_h
            x2 = (coords_xywh[:, 0] + widths) + pad_w
            y2 = (coords_xywh[:, 1] + heights) + pad_h
            raw_boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # Clean vectorized boundary clamping on host CPU memory
        raw_boxes[:, 0].clamp_(min=0, max=self.resize_w - 1)
        raw_boxes[:, 1].clamp_(min=0, max=self.resize_h - 1)
        raw_boxes[:, 2].clamp_(min=0, max=self.resize_w - 1)
        raw_boxes[:, 3].clamp_(min=0, max=self.resize_h - 1)

        return raw_boxes

    # CPU ------------------------------------------------
    def init_cpu_pipeline(self):
        self.prepare_cpu_pipeline()

    def allocate_cpu(self):
        self.device_index = "cpu"
        if not hasattr(
            self, "_pinned_small_frame"
        ):  # or self._pinned_small_frame.shape[:2] != (self.resize_h, self.resize_w):
            self._pinned_small_frame = np.zeros(
                (self.resize_h, self.resize_w, 3), dtype=np.uint8
            )
            self._pinned_fg_mask = np.zeros(
                (self.resize_h, self.resize_w), dtype=np.uint8
            )
            self._pinned_blurred_mask = np.zeros(
                (self.resize_h, self.resize_w), dtype=np.uint8
            )
            self._pinned_threshold_mask = np.zeros(
                (self.resize_h, self.resize_w), dtype=np.uint8
            )
            self._pinned_dilated_mask = np.zeros(
                (self.resize_h, self.resize_w), dtype=np.uint8
            )

        # Existing order and boolean validation maps
        self._filter_scratch_keep = torch.zeros(
            (self.max_cached_elements,), dtype=torch.bool, device=self.device_input
        )
        self._filter_order_scratch = torch.zeros(
            (self.max_cached_elements,), dtype=torch.long, device=self.device_input
        )

        # Persistent coordinate layers to absorb inner-loop tensor evaluations safely
        self._filter_scratch_x1 = torch.zeros(
            (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        )
        self._filter_scratch_y1 = torch.zeros(
            (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        )
        self._filter_scratch_x2 = torch.zeros(
            (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        )
        self._filter_scratch_y2 = torch.zeros(
            (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        )
        self._filter_scratch_ioa = torch.zeros(
            (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        )

        # pass
        self.resized_frame = np.zeros((3, self.resize_h, self.resize_w), dtype="uint8")
        # cv2.cuda.createContinuous(
        #     self.resize_h, self.resize_w, cv2.CV_8UC3
        # )

        self.fgMask = np.zeros(
            (self.resize_h, self.resize_w), dtype="uint8"
        )  # For resize

        if self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD == "and":
            self.prev_bkgd = np.ones(
                (self.resize_h, self.resize_w), dtype="uint8"
            )  # * 255
        else:
            self.prev_bkgd = np.zeros((self.resize_h, self.resize_w), dtype="uint8")

        # self.prev_bkgd = np.ones((self.resize_h, self.resize_w), dtype="uint8") * 255

        self.mask_history = deque(
            maxlen=self.config.BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE
        )
        self.mask_history.append(self.prev_bkgd)

    def prepare_cpu_pipeline(self):  # , method="mog2"):
        self.allocate_cpu()

        # Subtraction
        self.lr = self.config.BKGD_SUB_MOG2_LR
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=self.config.BKGD_SUB_MOG2_HISTORY,  # Clear ghosts of fast drones in ~2 seconds (2*fps)
            varThreshold=self.config.BKGD_SUB_MOG2_VARTHRESHOLD,  # High threshold to ignore "shimmer" and compression noise  # default 16
            detectShadows=self.config.BKGD_SUB_MOG2_DETECTSHADOWS,  # default True
        )
        # else:
        #     raise ValueError(f"Provided method ({method}) is not available.")

    def cleanup_cpu_v1(self):
        """
        Purges large 8K NumPy buffers and CPU-based AI resources.
        """
        self._pinned_small_frame = None
        self._pinned_fg_mask = None
        self._pinned_blurred_mask = None
        self._pinned_threshold_mask = None
        self._pinned_dilated_mask = None

        # Nullify specific class references to allow Garbage Collection
        self.executor = None
        self.clip_executor = None
        self.reader = None
        self.latest_processed_frame = None

        # Clear the Ping-Pong buffers (up to 200MB of RAM)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Explicitly nullify large arrays to trigger Garbage Collection
        self.resized_frame = None
        self.fgMask = None
        self.prev_bkgd = None

        # Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

    def apply_background_subtraction_cpu(
        self, motion_input, include_history=True, method="and"
    ):
        raw_mask = self.backSub.apply(
            motion_input, learningRate=self.config.BKGD_SUB_MOG2_LR
        )

        if include_history:
            # self.prev_bkgd = np.zeros((self.resize_h, self.resize_w), dtype="uint8")

            for m in list(self.mask_history):
                # Dilate the historical mask on CPU
                dilated = cv2.dilate(
                    m, self.dilate_kernel_for_enhanced_mask, iterations=1
                )

                if method == "or":
                    # Bitwise OR on CPU
                    cv2.bitwise_or(self.prev_bkgd, dilated, dst=self.prev_bkgd)
                else:
                    # Bitwise AND on CPU
                    cv2.bitwise_and(self.prev_bkgd, dilated, dst=self.prev_bkgd)

            self.mask_history.append(raw_mask)  # .copy())

            # if (
            #     self.prev_bkgd.max() != self.prev_bkgd.min()
            #     and self.prev_bkgd.max() > 0
            # ):
            #     combined_mask_bool = (self.fgMask > 0) | (self.prev_bkgd > 0)
            #     self.fgMask = combined_mask_bool.astype(np.uint8) * 255
            raw_mask = cv2.bitwise_or(raw_mask, self.prev_bkgd)
        return raw_mask

    def get_sf_cpu_rois(
        self, device_frame, overall_frame_num, max_candidates=100, limit_640=1280
    ):
        # is_debug = self.config.DEBUG_FLAG
        # test_mode = self.config.TEST_MODE
        debug_frame_limit = (
            self.debug_frame_limit
            if self.config.DEBUG_FLAG and self.debug_frame_limit > -1
            else -1
        )

        if overall_frame_num <= debug_frame_limit:
            stage_debug_dir = (
                self.result_dir / "debug_stages" / self._testMethodName / "roi_stages"
            )
            stage_debug_dir.mkdir(parents=True, exist_ok=True)
        # f_num = overall_frame_num

        if overall_frame_num <= debug_frame_limit:
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{overall_frame_num:04d}_stage1_src.jpg"),
                device_frame,
            )

        # target_w, target_h = self.resize_w, self.resize_h

        # Force a zero-allocation resize into our pre-allocated, sequential cache-line matrix buffer
        cv2.resize(
            device_frame,
            (self.resize_w, self.resize_h),
            dst=self._pinned_small_frame,
            interpolation=cv2.INTER_NEAREST,
        )
        # [STAGE 2 DEBUG] Check Downsampled Frame
        if overall_frame_num <= debug_frame_limit:
            cv2.imwrite(
                str(
                    stage_debug_dir / f"frame_{overall_frame_num:04d}_stage2_resize.jpg"
                ),
                self._pinned_small_frame,
            )

        # --- PHASE 2: STRIPPED SINGLE-THREAD BACKGROUND ARITHMETIC ---
        # Run background subtraction straight into our static memory address lane
        # We pass your configured learning rate (BKGD_SUB_MOG2_LR) to lock the temporal parameters
        # self._pinned_fg_mask = self.backSub.apply(
        #     self._pinned_small_frame,
        #     # dst=self._pinned_fg_mask,
        #     learningRate=self.config.BKGD_SUB_MOG2_LR,
        # )

        self._pinned_fg_mask = self.apply_background_subtraction_cpu(
            self._pinned_small_frame,
            include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
            method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
        )

        # [STAGE 5 DEBUG] Check Mask after History ORing/ANDing steps
        if overall_frame_num <= debug_frame_limit:
            cv2.imwrite(
                str(
                    stage_debug_dir
                    / f"frame_{overall_frame_num:04d}_stage5_history_mask.jpg"
                ),
                self._pinned_fg_mask,
            )

        # d_blurred = cv2.cuda.GpuMat(raw_mask.size(), cv2.CV_8UC1)
        # self._cuda_gaussian_filter.apply(raw_mask, d_blurred, stream=self.bgs_stream)
        ksize = (17, 17)
        self._pinned_blurred_mask = cv2.GaussianBlur(self._pinned_fg_mask, ksize, 0)

        # [STAGE 6 DEBUG]
        if overall_frame_num <= debug_frame_limit:
            cv2.imwrite(
                str(
                    stage_debug_dir / f"frame_{overall_frame_num:04d}_stage5b_blur.jpg"
                ),
                # str(stage_debug_dir / f"frame_{overall_frame_num:04d}_stage6_thresh_final_mask.jpg"),
                self._pinned_blurred_mask,
            )

        # 7. MORPHOLOGICAL TRANSFORMATIONS & BINARY FILTERS
        _, self._pinned_threshold_mask = cv2.threshold(
            self._pinned_blurred_mask,
            50,  # self.config.THRESHOLD_VALUE,
            self.config.THRESHOLD_MAX_VALUE,
            cv2.THRESH_BINARY,
        )

        # [STAGE 6 DEBUG] Check Binary Threshold Output
        if overall_frame_num <= debug_frame_limit:
            cv2.imwrite(
                str(
                    stage_debug_dir
                    / f"frame_{overall_frame_num:04d}_stage6_threshold.jpg"
                ),
                # str(stage_debug_dir / f"frame_{overall_frame_num:04d}_stage6_thresh_final_mask.jpg"),
                self._pinned_threshold_mask,
            )

        # Execute morphology dilation inline within our zero-allocation ring workspace [PDF: 0.1.18]
        cv2.dilate(
            self._pinned_threshold_mask,
            self.dilate_kernel,
            dst=self._pinned_dilated_mask,
            iterations=1,
        )
        # [STAGE 7 DEBUG] Check Final Dilated Output Mask
        if overall_frame_num <= debug_frame_limit:
            cv2.imwrite(
                str(
                    stage_debug_dir
                    / f"frame_{overall_frame_num:04d}_stage8_final_dilatemask.jpg"
                ),
                self._pinned_dilated_mask,
            )

        # inf_data = {"full_frame": frame, "mask": self._pinned_dilated_mask}

        # REGION OF INTEREST ANALYSIS
        # get cpu rois by area
        contours, _ = cv2.findContours(
            self._pinned_dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw_boxes = [
            list(cv2.boundingRect(c))
            for c in contours
            if cv2.contourArea(c) > self.min_contour_area
        ]

        if len(raw_boxes) < 1:
            return torch.empty((0, 4), device=self.device_input)

        if len(raw_boxes) > max_candidates:
            raw_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            raw_boxes = raw_boxes[:max_candidates]

        with self.compiled_no_grad_gate:
            # coords_xywh = np.array(raw_boxes_xywh, dtype=np.float32)
            coords_xywh = torch.tensor(
                raw_boxes, dtype=torch.float32, device=self.device_input
            )

            raw_boxes = self.cpu_roi_padding(coords_xywh, method="uniform")

            clean_640p = self.filter_contained_boxes(
                raw_boxes, overlap_thresh=self.config.ROI_CONTAINMENT_THRESH
            )

            if clean_640p.shape[0] < 1:
                # del raw_boxes_640p, clean_640p, raw_boxes_xywh  # , raw_boxes
                return torch.empty((0, 4), device=self.device_input)

            # Scale to 8K space
            # Sever the PyTorch graph graph entirely by detaching and casting to a standard numpy block
            bbs_full_res = (clean_640p * self.scales_tensor).detach()  # .clone()

            # Explicitly clear the intermediate tensor tensors to drop heap allocations to 0
            # del raw_boxes_640p, clean_640p, raw_boxes_xywh, scaled_output  # , raw_boxes

        return bbs_full_res

    def run_cpu_pipeline(
        self, device_frame, overall_frame_num, frame_in_clip_count=0, gt_boxes=None
    ):
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Get ROIs
        if self.timer_enabled:
            t_start = time.perf_counter()

        bbs_full_res = self.get_sf_cpu_rois(device_frame, overall_frame_num)

        if self.timer_enabled:
            metrics["sf_time"] = (time.perf_counter() - t_start) * 1000.0

        # metrics["bbs"] = bbs_full_res
        if bbs_full_res is not None:
            if torch.is_tensor(bbs_full_res):
                metrics["bbs"] = bbs_full_res.detach().cpu().numpy()
            elif isinstance(bbs_full_res, np.ndarray):
                metrics["bbs"] = bbs_full_res
        else:
            metrics["bbs"] = np.empty((0, 4), dtype=np.float32)

        merged, det_frame = self.format_bbs_and_frame_4_detection(
            bbs_full_res, device_frame
        )
        motion_detected = len(merged) > 0 if merged is not None else False
        metrics["batch_density"] = len(merged) if merged is not None else 0

        if (
            self.config.DEBUG_FLAG
            and overall_frame_num <= self.config.DEBUG_FRAME_LIMIT
        ):
            self.debug_save_mask(
                det_frame, overall_frame_num, rois=merged, gt_boxes=gt_boxes
            )

        if not self.config.DISABLE_DETECTION:
            # --- 3. MODEL INFERENCE TIMING BLOCK ---
            if self.timer_enabled:
                t_start = time.perf_counter()

            # Get Detection Metadata
            if self.config.DETECTION_TYPE != "motion":  # Object detection
                metadata, _ = self.get_detections(
                    det_frame,
                    frame_in_clip_count,
                    merged=merged,
                    thickness=self.config.THICKNESS,
                    device_input=self.config.device_input,
                )
                # num_objs = len(metadata.keys())
            else:  # Motion: Smart filtering results
                # 8k bb to 640
                metadata = self.motion2metadata(merged, self.frame_count_target)

            if self.timer_enabled:
                metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0

        del merged, bbs_full_res
        return metrics, metadata, det_frame, motion_detected

    def cleanup_gpu(self):
        """
        Explicitly releases all GPU-allocated memory, persistent tensors,
        and pinned cross-hardware streams to prevent VRAM leaks.
        """
        main_app_logger.info("[CLEANUP] Starting comprehensive GPU resource purge...")

        # 1. Clear the persistent VRAM lock tensor to allow full layout collection
        if hasattr(self, "_BaseObjectDetector__persistent_vram_lock"):
            self._BaseObjectDetector__persistent_vram_lock = None

        # 2. Iterate through and explicitly release all primitive OpenCV GpuMat layers
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, cv2.cuda.GpuMat):
                attr_value.release()
                setattr(self, attr_name, None)
                main_app_logger.info(f"  [x] Released GpuMat: {attr_name}")

        # 3. Explicitly nullify persistent PyTorch Tensors on GPU/CPU
        tensor_attributes = [
            "fixed_inference_batch",
            "gpu_float_staging",
            "scales_tensor",
            "pinned_cpu_xyxy",
            "pinned_cpu_clss",
            "pinned_cpu_confs",
            "static_canvas_scratch",
            "_labeled_scratch",
            "_filter_survival_mask",
            "_filter_scratch_keep",
            "_filter_order_scratch",
            "_filter_scratch_x1",
            "_filter_scratch_y1",
            "_filter_scratch_x2",
            "_filter_scratch_y2",
            "_filter_scratch_ioa",
        ]
        for attr in tensor_attributes:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # 4. Flush pre-allocated lists and buffer collections
        if hasattr(self, "gpu_buffer_pool"):
            for mat in self.gpu_buffer_pool:
                if isinstance(mat, cv2.cuda.GpuMat):
                    mat.release()
            self.gpu_buffer_pool = []

        if hasattr(self, "frame_buffer_pool"):
            self.frame_buffer_pool = []

        if hasattr(self, "mask_history"):
            self.mask_history.clear()

        # 5. Nullify models and framework engines to drop context weights
        self.model = None
        self.cached_predictor = None
        self.compiled_no_grad_gate = None
        if hasattr(self, "_raw_bounds_function"):
            self._raw_bounds_function = None

        # 6. Final unified hardware fence synchronization and allocator sweep
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        main_app_logger.info(
            "[CLEANUP] GPU resource tracking layers completely isolated."
        )

    def cleanup_cpu(self):
        """
        Purges large 8K NumPy buffers and host-side tracking structures.
        """
        main_app_logger.info("[CLEANUP] Purging host CPU variables...")

        cpu_buffers = [
            "_pinned_small_frame",
            "_pinned_fg_mask",
            "_pinned_blurred_mask",
            "_pinned_threshold_mask",
            "_pinned_dilated_mask",
            "resized_frame",
            "fgMask",
            "prev_bkgd",
        ]
        for attr in cpu_buffers:
            if hasattr(self, attr):
                setattr(self, attr, None)

        if hasattr(self, "mask_history"):
            self.mask_history.clear()

        # Clean background web workers and thread pool contexts safely
        self.executor = None
        self.clip_executor = None
        self.reader = None
        self.latest_processed_frame = None

        # Explicitly run standard garbage collection to drop unreferenced heap layers
        import gc

        gc.collect()
        main_app_logger.info("[CLEANUP] Host CPU workspace successfully cleared.")
