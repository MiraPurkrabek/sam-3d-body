# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob
import json

from time import time

import pyrootutils

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together, visualize_sample_animation
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

KPT_THR = 0.5

def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    human_detector, human_segmentor, fov_estimator = None, None, None

    predetected_bboxes = False
    predetected_keypoints = False
    predetected_masks = False
    if args.detected_bboxes_path != "":
        if not os.path.isfile(args.detected_bboxes_path):
            raise FileNotFoundError(
                f"Detected bounding boxes file not found: {args.detected_bboxes_path}"
            )

        with open(args.detected_bboxes_path, "r") as f:
            import json
            detected_bboxes = json.load(f)

            
            
            if isinstance(detected_bboxes, dict):
                if 'annotations' in detected_bboxes and 'images' in detected_bboxes:
                    print("Detected COCO-format detected_bboxes JSON file.")
                    # COCO format
                    image_id_to_img = {
                        img['id']: img for img in detected_bboxes['images']
                    }
                    image_name_to_bboxes = {}
                    image_name_to_keypoints = {}
                    image_name_to_masks = {}
                    image_name_to_scores = {}
                    for ann in detected_bboxes['annotations']:
                        image_name = image_id_to_img[ann['image_id']]['file_name']
                        bbox_xywh = np.array(ann['bbox'])
                        bbox_xyxy = np.array([
                            bbox_xywh[0],
                            bbox_xywh[1],
                            bbox_xywh[0] + bbox_xywh[2],
                            bbox_xywh[1] + bbox_xywh[3],
                        ])
                        if image_name not in image_name_to_bboxes:
                            image_name_to_bboxes[image_name] = []
                            image_name_to_scores[image_name] = []
                        image_name_to_bboxes[image_name].append(bbox_xyxy)
                        image_name_to_scores[image_name].append(ann.get('score', -1.0))

                        if 'keypoints' in ann:
                            kpts = np.array(ann['keypoints']).reshape(-1, 3)
                            # kpts[kpts[:, 2] < KPT_THR, :] = [0, 0, -2]
                            if image_name not in image_name_to_keypoints:
                                image_name_to_keypoints[image_name] = []
                            image_name_to_keypoints[image_name].append(kpts)
                            predetected_keypoints = True

                        if 'segmentation' in ann:
                            poly_mask = ann['segmentation']
                            img_h = image_id_to_img[ann['image_id']]['height']
                            img_w = image_id_to_img[ann['image_id']]['width']
                            binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            if isinstance(poly_mask, list):
                                # Polygon format
                                for poly in poly_mask:
                                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                                    cv2.fillPoly(binary_mask, [pts], color=1)
                            else:
                                # 'poly_mask' is in fact in RLE format
                                from pycocotools import mask as maskUtils
                                binary_mask = maskUtils.decode(poly_mask)
                            segm = binary_mask.astype(bool)

                            if image_name not in image_name_to_masks:
                                image_name_to_masks[image_name] = []
                            image_name_to_masks[image_name].append(segm)
                            predetected_masks = True


                else:
                    raise ValueError("Unsupported detected_bboxes format in JSON file. Not a list but not a COCO-format either")
            elif isinstance(detected_bboxes, list):
                print("Detected list-format detected_bboxes JSON file.")
                image_name_to_bboxes = {}
                for det in detected_bboxes:
                    image_name = det['image_path']
                    bbox_xywh = np.array(det['bbox'])
                    bbox_xyxy = np.array([
                        bbox_xywh[0],
                        bbox_xywh[1],
                        bbox_xywh[0] + bbox_xywh[2],
                        bbox_xywh[1] + bbox_xywh[3],
                    ])
                    # score = det.get('bbox_score', 1.0)
                    if image_name not in image_name_to_bboxes:
                        image_name_to_bboxes[image_name] = []
                    image_name_to_bboxes[image_name].append(bbox_xyxy)
            else:
                raise ValueError("Unsupported detected_bboxes format in JSON file. Not a list or dict.")

        predetected_bboxes = True

    if args.detector_name and not predetected_bboxes:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    
    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)


    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )
    if len(images_list) == 0:
        print(f"[WARNING] No images found in {args.image_folder}. Supported formats: {image_extensions}")
        return


    if human_detector is not None:
        print(f"Initialized human detector: {args.detector_name}")

        image_name_to_bboxes = {}
        image_name_to_scores = {}

        predetected_bboxes = True  # Set this to True to skip detection in the main loop since we already have detections from JSON

        for image_path in tqdm(images_list, desc="Running human detection on images", ascii=True):
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            det_bboxes, det_scores = human_detector.run_human_detection(img)

            image_name_to_bboxes[image_name] = det_bboxes
            image_name_to_scores[image_name] = det_scores


            
        del human_detector  # Free up memory if detector is not needed for the next steps


    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,  # Keep detector as None to use less CUDA memory
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    time_accumulator = np.array([])



    keypoints_results = []
    for image_path in tqdm(images_list, desc="Processing images", ascii=True):
        image_name = os.path.basename(image_path)

        # selected_images = [
        #     "000076.jpg",
        #     "000043.jpg",
        #     "000283.jpg",
        #     "000246.jpg",
        #     "000185.jpg",
        #     "000210.jpg",
        #     "000174.jpg",
        #     "000185.jpg",
        #     "000257.jpg",
        #     "002909.jpg",
        # ]
        # if image_name not in selected_images:
        #     continue

        bboxes = None
        masks = None
        scores = None
        if predetected_bboxes:
            bboxes = np.array(image_name_to_bboxes.get(os.path.basename(image_path), []))
            scores = np.array(image_name_to_scores.get(os.path.basename(image_path), []))
            
            # Masks can be used only with bboxes
            if predetected_masks and args.use_mask:
                masks = np.array(image_name_to_masks.get(os.path.basename(image_path), []))
        
        keypoints = None
        if predetected_keypoints and args.use_keypoints:
            keypoints = np.array(
                image_name_to_keypoints.get(os.path.basename(image_path), [])
            )

        print("="*40)
        print("Processing image:", image_name)
        start_time = time()
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            inference_type="full",
            use_mask=predetected_masks,
            bboxes=bboxes,
            keypoints=keypoints,
            masks=masks,
        )
        end_time = time()
        print(f"Processing time for {image_name}: {end_time - start_time:.2f} seconds")
        time_accumulator = np.append(time_accumulator, end_time - start_time)
        print(f"Average processing time: {np.mean(time_accumulator):.2f} seconds")
        print(f"FPS: {1.0 / np.mean(time_accumulator):.2f}")
        print(f"Median processing time: {np.median(time_accumulator):.2f} seconds")


        for out in outputs:
            ann = {
                'image_path': os.path.basename(image_path),
                'bbox': out['bbox'].astype(float).flatten().tolist(),
                'keypoints_SAM3DBody': out['pred_keypoints_2d'].astype(float).flatten().tolist(),
                'bbox_score': np.array(out.get('bbox_score', 0.0)).astype(float).item(),
                'id': len(keypoints_results),
            }
            keypoints_results.append(ann)

        if len(outputs) == 0:
            continue

        img = cv2.imread(image_path)
        if scores is not None:
            for s, out in zip(scores, outputs):
                out['bbox_score'] = s
            
        rend_img = visualize_sample_together(img, outputs, estimator.faces, masks, keypoints)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )

        if args.rotation_animation:
            rend_animation = visualize_sample_animation(img, outputs, estimator.faces)
            os.makedirs(
                os.path.join(output_folder, "animations", os.path.basename(image_path)[:-4]), exist_ok=True
            )
            for i, out in enumerate(rend_animation):
                cv2.imwrite(
                    f"{output_folder}/animations/{os.path.basename(image_path)[:-4]}/frame_{i:04d}.jpg",
                    out,
                )

    # Save keypoints results
    import json
    with open(os.path.join(output_folder, "keypoints_results.json"), "w") as f:
        json.dump(keypoints_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--use_keypoints",
        action="store_true",
        default=False,
        help="Use kpts-conditioned prediction (keypoints are automatically generated from bbox)",
    )
    parser.add_argument(
        "--detected_bboxes_path",
        default="",
        type=str,
        help="Path to detected bounding boxes (optional, in case you want to use your own detections)",
    )
    parser.add_argument(
        "--rotation_animation",
        action="store_true",
        default=False,
        help="Generate rotation animation for each frame",
    )
    args = parser.parse_args()

    main(args)
