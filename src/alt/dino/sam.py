import cv2
import torch
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, r"D:\Models\GroundingDINO")
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

import groundingdino.models.GroundingDINO.ms_deform_attn as msda

class _PyMSDA:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index,
              sampling_locations, attention_weights, im2col_step):
        return msda.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

msda.MultiScaleDeformableAttnFunction = _PyMSDA

sys.path.insert(0, r"D:\Models\sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_CONFIG  = r"D:\Models\sam2\sam2\configs\sam2\sam2_hiera_l.yaml"
SAM2_CKPT    = r"D:\Models\sam2_hiera_large.pt"


sam2_model  = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Used for the dense-scene path (generate_road_mask). Built once, reused across calls.
mask_generator = SAM2AutomaticMaskGenerator(
    sam2_model,
    points_per_side=64,        # denser grid -> catches thin/small road segments
    pred_iou_thresh=0.7,
    stability_score_thresh=0.85,
    min_mask_region_area=50,   # drop tiny noise masks
)


def draw_boxes(image_np, boxes_xyxy, phrases_logits, out_path="dino_boxes.png"):
    img_boxes = image_np.copy()
    for (box, (phrase, logit)) in zip(boxes_xyxy, phrases_logits):
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(img_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{phrase} {logit:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_boxes, (x0, y0 - th - 6), (x0 + tw + 4, y0), (0, 255, 0), -1)
        cv2.putText(img_boxes, label, (x0 + 2, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR))
    return img_boxes


# ---------------------------------------------------------------------------
# Path B: SAM automatic mask generation (no DINO box) + shape filtering.
# One SAM encoder pass on the whole image, many small candidate masks out,
# then keep only the ones that look road-shaped (long/thin, low fill ratio,
# not too large). This is the one to use for dense suburban/urban tiles where
# Path A's single DINO box swallows most of the image.
# ---------------------------------------------------------------------------
def generate_road_mask(image_path, min_elongation=3.0, max_area_frac=0.05):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    anns = mask_generator.generate(image_np)  # one SAM encoder pass, many masks out

    road_mask = np.zeros((h, w), dtype=bool)
    kept = []

    for ann in anns:
        m = ann["segmentation"]
        area = ann["area"]

        # reject masks that are too big to be a road segment (roofs/lots/whole-image blobs)
        if area > max_area_frac * h * w:
            continue

        # shape filter: roads are long and thin -> low width/length ratio
        ys, xs = np.where(m)
        if len(xs) < 10:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bbox_w, bbox_h = x1 - x0 + 1, y1 - y0 + 1
        long_side, short_side = max(bbox_w, bbox_h), max(min(bbox_w, bbox_h), 1)
        elongation = long_side / short_side

        # fill ratio: a road strip fills a much smaller fraction of its bbox
        # than a blob (e.g. rooftop) does
        fill_ratio = area / (bbox_w * bbox_h)

        if elongation >= min_elongation or fill_ratio < 0.35:
            road_mask |= m
            kept.append(ann)

    return image_np, road_mask, kept


def save_overlay(image_np, mask, out_path):
    overlay = image_np.copy()
    overlay[mask] = [255, 0, 0]
    blended = cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)
    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    image_path = r"D:\osm_data\Massachusetts\tiff\test\20878930_15.tiff"
    img, mask, kept = generate_road_mask(image_path)
    print(f"kept {len(kept)} road-like masks")
    save_overlay(img, mask, "mask_overlay_sam_auto.png")