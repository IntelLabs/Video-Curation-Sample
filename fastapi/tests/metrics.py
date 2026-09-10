import torch


class DeviceAgnosticOnTheFlyEvaluator:
    def __init__(self, device="cpu"):
        # device will be "cpu" or "cuda" / "gpu" depending on what pytest passes
        self.device = "cuda" if "gpu" in str(device).lower() else "cpu"

        iou_range = [0.10, 0.95]
        self.num_steps = int(round(((iou_range[1] - iou_range[0]) / 0.05) + 1, 0))
        self.iou_thresholds = torch.linspace(
            iou_range[0], iou_range[1], self.num_steps, device=self.device
        )

        self.global_scores = []
        self.global_matches = []
        self.total_gt_count = 0
        self.total_gt_hit_lenient = 0  # GT boxes hit by ANY prediction with IoU >= 0.10

    @torch.inference_mode()
    def update_frame(self, pred_boxes, pred_scores, gt_boxes):
        """
        Accepts bounding boxes and scores as arrays or tensors.
        Automatically migrates inputs to the targeted CPU or GPU sandbox backend.
        """
        # Ensure all incoming payloads are safely handled as PyTorch tensors on the target device
        pred_boxes = torch.as_tensor(
            pred_boxes, dtype=torch.float32, device=self.device
        )
        pred_scores = torch.as_tensor(
            pred_scores, dtype=torch.float32, device=self.device
        )
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32, device=self.device)

        # Filter invalid ground truth placeholders (-1)
        if gt_boxes.numel() > 0:
            valid_mask = ~torch.all(gt_boxes == -1, dim=1)
            gt_boxes = gt_boxes[valid_mask]

        num_gt_in_frame = gt_boxes.shape[0]
        self.total_gt_count += num_gt_in_frame
        if pred_boxes.shape[0] == 0:
            return

        # Sort frame predictions by confidence score descending
        pred_scores, sort_idx = torch.sort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]

        # Allocate matching registers for this frame
        frame_matches = torch.zeros(
            (pred_boxes.shape[0], self.num_steps), dtype=torch.int32, device=self.device
        )

        if num_gt_in_frame > 0:
            # Vectorized Broadcast IoU Calculation
            preds_exp = pred_boxes.unsqueeze(1)  # (N, 1, 4)
            gts_exp = gt_boxes.unsqueeze(0)  # (1, M, 4)

            x_min = torch.maximum(preds_exp[..., 0], gts_exp[..., 0])
            y_min = torch.maximum(preds_exp[..., 1], gts_exp[..., 1])
            x_max = torch.minimum(preds_exp[..., 2], gts_exp[..., 2])
            y_max = torch.minimum(preds_exp[..., 3], gts_exp[..., 3])

            inter_area = torch.clamp(x_max - x_min, min=0) * torch.clamp(
                y_max - y_min, min=0
            )
            pred_areas = (preds_exp[..., 2] - preds_exp[..., 0]) * (
                preds_exp[..., 3] - preds_exp[..., 1]
            )
            gt_areas = (gts_exp[..., 2] - gts_exp[..., 0]) * (
                gts_exp[..., 3] - gts_exp[..., 1]
            )
            union_area = pred_areas + gt_areas - inter_area

            iou_matrix = torch.where(union_area > 0, inter_area / union_area, 0.0)

            if iou_matrix.numel() > 0:
                # Get the highest IoU any prediction achieved for each GT box
                max_ious_per_gt, _ = torch.max(iou_matrix, dim=0)
                # If a GT box was overlapped by at least 10%, consider it "found"
                self.total_gt_hit_lenient += torch.sum(max_ious_per_gt >= 0.10).item()

            # Match targets per threshold layer
            for th_idx, th in enumerate(self.iou_thresholds):
                matched_gts = torch.zeros(
                    num_gt_in_frame, dtype=torch.bool, device=self.device
                )
                for p_idx in range(pred_boxes.shape[0]):
                    row_ious = iou_matrix[p_idx]
                    # Mask out already matched ground truths
                    row_ious = torch.where(~matched_gts, row_ious, -1.0)

                    best_iou, best_gt_idx = torch.max(row_ious, dim=0)

                    if best_iou >= th and best_gt_idx != -1:
                        if row_ious[best_gt_idx] >= th:  # Extra validation gate
                            frame_matches[p_idx, th_idx] = 1
                            matched_gts[best_gt_idx] = True

        # Track only tiny matrix shards instead of bulky bounding box contexts
        # self.global_scores.append(pred_scores)
        # self.global_matches.append(frame_matches)

        # Force detach data fields completely out of the PyTorch hardware execution graph
        # and cast down to standard CPU lists before storing them over long durations
        if hasattr(pred_scores, "detach"):
            scores_list = pred_scores.detach().cpu().tolist()
        else:
            scores_list = list(pred_scores)

        if hasattr(frame_matches, "detach"):
            matches_list = frame_matches.detach().cpu().tolist()
        else:
            matches_list = list(frame_matches)

        # EXTEND to keep the list flat and completely avoid ragged dimensions
        self.global_scores.extend(scores_list)
        self.global_matches.extend(matches_list)

        del pred_scores, pred_boxes, gt_boxes

    @torch.inference_mode()
    def compute_final_metrics(self):
        """Processes final precision/recall curves natively on the configured hardware device."""
        if len(self.global_scores) == 0:
            return {
                "mAP_10": 0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "mAP_10_95": 0.0,
                "Precision_10": 0.0,
                "Recall_10": 0.0,
                "F1_Score_10": 0.0,
                "Precision_50": 0.0,
                "Recall_50": 0.0,
                "F1_Score_50": 0.0,
                "Max_Recall_IoU_10": 0.0,
            }

        # Flatten list of partial tensors into global timeline matrices
        # scores = torch.cat(self.global_scores, dim=0)
        # matches = torch.cat(self.global_matches, dim=0)

        # Flatten list of pre-detached primitive layouts into clean standalone timeline tensors
        scores = torch.tensor(
            self.global_scores, dtype=torch.float32, device=self.device
        ).view(-1)
        matches = torch.tensor(
            self.global_matches, dtype=torch.int32, device=self.device
        ).view(-1, self.num_steps)

        # Global Sort across the whole timeline by confidence scores
        _, global_sort_idx = torch.sort(scores, descending=True)
        matches = matches[global_sort_idx]

        aps = []

        # Track dictionary mapping for multi-threshold statistical processing
        # Key: Threshold Value (float) -> Values: (Precision, Recall)
        target_metrics = {
            0.10: {"precision": 0.0, "recall": 0.0},
            0.50: {"precision": 0.0, "recall": 0.0},
        }

        for th_idx in range(self.num_steps):
            current_th = float(f"{self.iou_thresholds[th_idx].item():.2f}")

            tp_cum = torch.cumsum(matches[:, th_idx], dim=0)
            fp_cum = torch.cumsum(1 - matches[:, th_idx], dim=0)

            precisions = tp_cum / (tp_cum + fp_cum)
            if self.total_gt_count > 0:
                recalls = tp_cum / self.total_gt_count
            else:
                recalls = torch.zeros_like(tp_cum)

            # Vectorized 101-point COCO AP Interpolation Layout
            mpre = torch.cat(
                [
                    torch.tensor([0.0], device=self.device),
                    precisions,
                    torch.tensor([0.0], device=self.device),
                ]
            )
            mrec = torch.cat(
                [
                    torch.tensor([0.0], device=self.device),
                    recalls,
                    torch.tensor([1.0], device=self.device),
                ]
            )

            # Enforce strict decreasing precision monotonicity
            mpre = torch.flip(
                torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values, dims=[0]
            )

            recall_thresholds = torch.linspace(0.0, 1.0, 101, device=self.device)
            inds = torch.searchsorted(mrec, recall_thresholds, side="left")
            inds = torch.clamp(inds, max=mpre.shape[0] - 1)

            ap = torch.sum(mpre[inds]) / 101.0
            aps.append(ap.item())

            # if th_idx == 0:
            #     final_precision_10 = (
            #         precisions[-1].item() if precisions.numel() > 0 else 0.0
            #     )
            #     final_recall_10 = recalls[-1].item() if recalls.numel() > 0 else 0.0

            # Dynamically extract and assign metrics if threshold matches target
            if current_th in target_metrics:
                target_metrics[current_th]["precision"] = (
                    precisions[-1].item() if precisions.numel() > 0 else 0.0
                )
                target_metrics[current_th]["recall"] = (
                    recalls[-1].item() if recalls.numel() > 0 else 0.0
                )

        # f1_10 = (
        #     (2 * final_precision_10 * final_recall_10)
        #     / (final_precision_10 + final_recall_10)
        #     if (final_precision_10 + final_recall_10) > 0
        #     else 0.0
        # )

        # Calculate F1 Score harmonic balances
        p10, r10 = target_metrics[0.10]["precision"], target_metrics[0.10]["recall"]
        f1_10 = (2 * p10 * r10) / (p10 + r10) if (p10 + r10) > 0 else 0.0

        p50, r50 = target_metrics[0.50]["precision"], target_metrics[0.50]["recall"]
        f1_50 = (2 * p50 * r50) / (p50 + r50) if (p50 + r50) > 0 else 0.0

        max_recall_10 = (
            (self.total_gt_hit_lenient / self.total_gt_count)
            if self.total_gt_count > 0
            else 0.0
        )

        # Robust close-approximation lookups bypassing basic Python indexing limitations
        map10_idx = torch.argmin(torch.abs(self.iou_thresholds - 0.10)).item()
        map50_idx = torch.argmin(torch.abs(self.iou_thresholds - 0.50)).item()
        map75_idx = torch.argmin(torch.abs(self.iou_thresholds - 0.75)).item()

        return {
            "mAP_10": aps[map10_idx],
            "mAP_50": aps[map50_idx],  # mAP at 0.50 IoU
            "mAP_75": aps[map75_idx],  # mAP at 0.75 IoU
            "mAP_10_95": sum(aps)
            / float(self.num_steps),  # Averaged mAP over all 18 threshold layers
            "Precision_10": p10,
            "Recall_10": r10,
            "F1_Score_10": f1_10,
            "Precision_50": p50,
            "Recall_50": r50,
            "F1_Score_50": f1_50,
            "Max_Recall_IoU_10": max_recall_10,
        }
