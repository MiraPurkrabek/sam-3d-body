# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def _coco_to_prompt(self, coco_kps: np.ndarray, batch: dict, score_thresh: float = 0.3) -> torch.Tensor:
        """
        Convert COCO-format keypoints to model prompt tensor.

        Args:
            coco_kps: numpy array in shape (P, 17, 3) or (B, P, 17, 3) with (x, y, score)
            batch: prepared batch (already moved to device)
            score_thresh: minimum score to consider a keypoint as present

        Returns:
            torch.Tensor of shape (B*P, N, 3) where N is the max number of prompts per person.
            Last column is the label (MHR keypoint index), -2 indicates dummy.
        """
        device = batch["img"].device

        # Determine batch and person counts
        bs = int(batch["img"].shape[0])
        # number of person crops
        num_person = int(batch["img"].shape[1])

        kp = np.array(coco_kps)
        # Accept shapes: (P,17,3) for single image or (B,P,17,3)
        if kp.ndim == 3:
            # expand to (1, P, 17, 3)
            kp = kp[None]
        if kp.shape[0] != bs or kp.shape[1] != num_person:
            # try broadcasting single-image keypoints to batch
            if kp.shape[0] == 1 and bs > 1:
                kp = np.repeat(kp, bs, axis=0)
            else:
                raise ValueError("Provided keypoints shape does not match batch/img layout")

        # COCO -> MHR mapping for the 17 standard COCO joints
        coco_to_mhr = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 62,  # left_wrist
            10: 41,  # right_wrist
            11: 9,  # left_hip
            12: 10,  # right_hip
            13: 11,  # left_knee
            14: 12,  # right_knee
            15: 13,  # left_ankle
            16: 14,  # right_ankle
        }

        # Convert to torch and move to device
        kp_t = torch.from_numpy(kp).to(device=device, dtype=torch.float32)

        # Flatten persons similar to model._flatten_person
        flat_affine = self.model._flatten_person(batch["affine_trans"])  # (B*P, 2, 3)
        flat_img_size = self.model._flatten_person(batch["img_size"]).unsqueeze(1).to(
            device
        )  # (B*P, 1, 2)

        B = bs * num_person
        kp_flat = kp_t.view(B, 17, 3)

        # Convert full-image (x,y) to crop coordinates and normalize to [-0.5,0.5]
        ones = torch.ones((B, 17, 1), device=device)
        xy1 = torch.cat([kp_flat[:, :, :2], ones], dim=-1)  # (B,17,3)
        # affine: (B,2,3) -> transpose to (B,3,2) for matmul
        crop_xy = torch.matmul(xy1, flat_affine.permute(0, 2, 1))  # (B,17,2)
        norm_xy = crop_xy / flat_img_size - 0.5  # (B,17,2)

        prompts_per_person = []
        for i in range(B):
            cur = []
            scores = kp_flat[i, :, 2]
            for coco_idx, mhr_idx in coco_to_mhr.items():
                if scores[coco_idx] > score_thresh:
                    # Use crop-normalized coords in [-0.5,0.5], then shift to [0,1]
                    x = (norm_xy[i, coco_idx, 0] + 0.5).clamp(0.0, 1.0)
                    y = (norm_xy[i, coco_idx, 1] + 0.5).clamp(0.0, 1.0)
                    label = torch.tensor(float(mhr_idx), device=device)
                    p = torch.stack((x, y, label))
                    cur.append(p)
            if len(cur) == 0:
                # create a dummy prompt so model behavior is unchanged
                d = torch.tensor([0.0, 0.0, -2.0], device=device)
                prompts_per_person.append(d.unsqueeze(0))
            else:
                prompts_per_person.append(torch.stack(cur, dim=0))

        # Pad to equal length
        max_len = max(p.shape[0] for p in prompts_per_person)
        out = torch.full((B, max_len, 3), 0.0, device=device, dtype=torch.float32)
        for i, p in enumerate(prompts_per_person):
            out[i, : p.shape[0]] = p
            out[i, p.shape[0]:, -1] = -2.0

        return out

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
        keypoints: Optional[np.ndarray] = None,
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            bbox_scores = np.ones(len(boxes), dtype=np.float32)
            print("Using provided bboxes:", boxes)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes, bbox_scores = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        # If external COCO-format keypoints were provided, convert them to
        # the model's prompt format and run a keypoint-conditioned pass.
        # keypoints is expected in COCO order (17 joints): (P, 17, 3) or
        # (B, P, 17, 3) with (x, y, score). Coordinates are full-image pixels.
        if keypoints is not None:
            try:
                # Ensure prompts are a torch.Tensor on the same device as batch
                kp_prompt = self._coco_to_prompt(keypoints, batch)
            except Exception as exc:
                print(f"[SAM3DBodyEstimator] Failed to convert COCO keypoints to prompts: {exc}")
            else:
                # run_inference returns different outputs for 'full' vs others
                if inference_type == "full":
                    (
                        pose_output,
                        batch_lhand,
                        batch_rhand,
                        lhand_output,
                        rhand_output,
                    ) = outputs
                else:
                    pose_output = outputs

                # Run the keypoint prompt refinement
                print("Refining prediction with provided keypoint prompts...")
                # print(kp_prompt)
                updated_pose_output, _ = self.model.run_keypoint_prompt(
                    batch, pose_output, kp_prompt
                )

                if inference_type == "full":
                    outputs = (
                        updated_pose_output,
                        batch_lhand,
                        batch_rhand,
                        lhand_output,
                        rhand_output,
                    )
                else:
                    outputs = updated_pose_output
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "bbox_score": bbox_scores[idx] if bbox_scores is not None else None,
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out
