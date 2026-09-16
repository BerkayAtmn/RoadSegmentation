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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- adjust these 4 paths to match where your files actually are ---
GDINO_CONFIG = r"D:\Models\GroundingDINO\groundingdino\config\GroundingDINO_SwinB_cfg.py"
GDINO_CKPT   = r"D:\Models\groundingdino_swinb_cogcoor.pth"
SAM2_CONFIG  = r"D:\Models\sam2\sam2\configs\sam2\sam2_hiera_l.yaml"
SAM2_CKPT    = r"D:\Models\sam2_hiera_large.pt"

gdino_model = load_model(GDINO_CONFIG, GDINO_CKPT, device=DEVICE)
sam2_model  = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

def detect_and_segment(image_path, text_prompt="green areas",
                        box_threshold=0.25, text_threshold=0.25):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    boxes, logits, phrases = predict(
        model=gdino_model, image=image_tensor, caption=text_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold, device=DEVICE,
    )

    if len(boxes) == 0:
        return image_np, np.zeros((h, w), dtype=bool), [], np.zeros((0, 4))

    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    boxes_xyxy = boxes_xyxy.cpu().numpy()

    sam2_predictor.set_image(image_np)
    masks = [sam2_predictor.predict(box=b, multimask_output=False)[0][0] for b in boxes_xyxy]
    combined_mask = np.any(np.stack(masks), axis=0)
    return image_np, combined_mask, list(zip(phrases, logits.tolist())), boxes_xyxy

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

def run_16x16_tiled(image_path, text_prompt, **kwargs):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    H, W = image_np.shape[:2]

    full_mask = np.zeros((H, W), dtype=bool)
    tile = 100

    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y1, x1 = min(y + tile, H), min(x + tile, W)
            crop = Image.fromarray(image_np[y:y1, x:x1])
            crop_path = "tmp_crop.png"
            crop.save(crop_path)

            try:
                _, mask, dets, boxes = detect_and_segment(crop_path, text_prompt=text_prompt, **kwargs)
                full_mask[y:y1, x:x1] |= mask
            except Exception as e:
                continue
    return image_np, full_mask

if __name__ == "__main__":
    img, mask = run_16x16_tiled(
        #r"D:\osm_data\DeepGlobeRoadExtractionDataset\test\62796_sat.jpg",
        r"D:\osm_data\SatelitteRoadSegmentation\Massachusetts\tiff\test\20878930_15.tiff",
        text_prompt="overlook roads"
    )

    overlay = img.copy()
    overlay[mask] = [255, 0, 0]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite("mask_overlay_16x16.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))