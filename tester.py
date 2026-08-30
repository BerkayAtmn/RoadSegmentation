# test_viewer.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from road_dataset import RoadImageDataset
from inference import sliding_window_predict

root_massachusetts = r"C:\Dev\deeplearning\data\massachusetts-roads-dataset"
root_deepglobe = r"C:\Dev\deeplearning\data\deepglobe"
checkpoint_path = r"C:\Dev\deeplearning\trained_models\roadseg_merged_dice_best.pt"

# set to False to skip the plt.show() windows and just crunch stats
SHOW_PLOTS = True

# Tweak this! Lowering it (e.g., 0.35 - 0.45) often improves Recall and IoU for imbalanced classes.
THRESHOLD = 0.4

# Same tiling as training-time validation (see inference.py / notebook.ipynb) — full
# tiles get merged predictions, not a single crop or a single whole-image forward pass.
WINDOW, STRIDE = 512, 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Overlay colors (RGB, 0-1) — picked for sharp contrast against satellite imagery.
PRED_COLOR = (1.0, 0.0, 0.0)  # red
GT_COLOR = (0.0, 1.0, 1.0)  # cyan
OVERLAY_ALPHA = 0.6


def compute_confusion_counts(pred, gt, positive_class=1):
    """
    pred, gt: 2D numpy arrays of {0,1} (or any int labels), same shape.
    Returns TP, FP, FN, TN counts for the given positive_class (binary: road=1).
    """
    pred_pos = (pred == positive_class)
    gt_pos = (gt == positive_class)

    tp = np.logical_and(pred_pos, gt_pos).sum()
    fp = np.logical_and(pred_pos, ~gt_pos).sum()
    fn = np.logical_and(~pred_pos, gt_pos).sum()
    tn = np.logical_and(~pred_pos, ~gt_pos).sum()
    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    """Standard binary segmentation metrics from confusion-matrix counts."""
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def gpu_normalize(images_u8, device):
    """images_u8: (B,3,H,W) uint8. Returns ImageNet-normalized float32 on `device`,
    matching the ToDtype(float32, scale=True) + Normalize the model was trained with."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    images = images_u8.to(device).float().div_(255.0)
    return images.sub_(mean).div_(std)


def predict_tile(model, image_u8, device):
    """image_u8: (3,H,W) uint8, full tile. Returns (H,W) road probability, on `device`."""
    images = gpu_normalize(image_u8.unsqueeze(0), device)
    probs = sliding_window_predict(model, images, window=WINDOW, stride=STRIDE)
    return probs[0, 0]


def evaluate_dataset(model, dataset, device, label, out_csv=None):
    """Run sliding-window inference over every tile in `dataset` and report average
    stats two ways: global (micro, pooled over every pixel in the whole set) and
    per-tile (macro, mean of each tile's own metrics)."""
    total_tp = total_fp = total_fn = total_tn = 0
    records = []

    for idx in range(len(dataset)):
        image_u8, mask_u8 = dataset[idx]
        image_name = os.path.basename(dataset.pairs[idx][0])

        with torch.no_grad():
            probs = predict_tile(model, image_u8, device)
        pred = (probs > THRESHOLD).cpu().numpy().astype(np.int64)
        gt = mask_u8.squeeze(0).numpy().astype(np.int64)

        tp, fp, fn, tn = compute_confusion_counts(pred, gt, positive_class=1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        records.append({"image": image_name, **metrics_from_counts(tp, fp, fn, tn)})

    overall = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    df = pd.DataFrame(records)
    macro = df[["accuracy", "precision", "recall", "f1", "iou"]].mean().to_dict()

    print(f"\n{'=' * 60}\n{label}: {len(dataset)} test tiles\n{'=' * 60}")
    print("Global (micro-averaged, pooled over every pixel):")
    for k, v in overall.items():
        print(f"  {k:10s}: {v * 100:6.2f}%")
    print("Per-tile (macro-averaged):")
    for k, v in macro.items():
        print(f"  {k:10s}: {v * 100:6.2f}%")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Per-tile metrics saved to: {out_csv}")

    return overall, macro


def overlay_mask(image_rgb01, mask, color, alpha=OVERLAY_ALPHA):
    """image_rgb01: (H,W,3) float in [0,1]. mask: (H,W) bool. color: (r,g,b) in [0,1]."""
    overlay = image_rgb01.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.float32)
    return overlay


def visualize_massachusetts(model, dataset, device):
    """For each Massachusetts test tile, show the predicted road mask over the
    satellite image and the ground-truth road mask over the same image, each in
    its own sharp color, side by side."""
    for idx in range(len(dataset)):
        image_u8, mask_u8 = dataset[idx]
        image_name = os.path.basename(dataset.pairs[idx][0])

        with torch.no_grad():
            probs = predict_tile(model, image_u8, device)
        pred = (probs > THRESHOLD).cpu().numpy().astype(bool)
        gt = mask_u8.squeeze(0).numpy().astype(bool)

        img_disp = image_u8.permute(1, 2, 0).numpy().astype(np.float32) / 255.0

        pred_overlay = overlay_mask(img_disp, pred, PRED_COLOR)
        gt_overlay = overlay_mask(img_disp, gt, GT_COLOR)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(pred_overlay)
        axes[0].set_title(f"{image_name}\nPredicted roads (red)")
        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"{image_name}\nGround-truth roads (cyan)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()  # blocks until you close the window, then moves to next tile


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()

    ckpt_dir = os.path.dirname(checkpoint_path)

    # --- Massachusetts: real, labeled test split ---
    mass_test = RoadImageDataset(root_massachusetts, "test", transform=None)
    evaluate_dataset(
        model, mass_test, device, "Massachusetts",
        out_csv=os.path.join(ckpt_dir, "test_metrics_massachusetts.csv"),
    )

    # --- DeepGlobe: its own test split ships without masks, so there's nothing to
    # score against. Fall back to the same held-out fraction of train the notebook
    # carves off as its val set (same seed) — not a truly independent test set, but
    # the only labeled DeepGlobe data outside of training. ---
    dg_test = RoadImageDataset(root_deepglobe, "val", transform=None, val_fraction=0.1, seed=42)
    evaluate_dataset(
        model, dg_test, device, "DeepGlobe (val carve-out; test split has no masks)",
        out_csv=os.path.join(ckpt_dir, "test_metrics_deepglobe.csv"),
    )

    if SHOW_PLOTS:
        print(f"\nVisualizing {len(mass_test)} Massachusetts tiles — close each plot window to see the next.")
        visualize_massachusetts(model, mass_test, device)


if __name__ == "__main__":
    main()
