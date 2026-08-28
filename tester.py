# test_viewer.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
import segmentation_models_pytorch as smp

root = r"C:\Dev\deeplearning\data\massachusetts-roads-dataset"
checkpoint_path = r"C:\Dev\deeplearning\trained_models\roadseg_epoch18_Dice_lr1e-4.pt"

# set to False to skip the plt.show() windows and just crunch stats
SHOW_PLOTS = True

# Tweak this! Lowering it (e.g., 0.35 - 0.45) often improves Recall and IoU for imbalanced classes.
THRESHOLD = 0.5 

# Updated to use v2 transforms to match your training pipeline
image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])


def pad_to_multiple(tensor, multiple=32):
    """Pad a CHW or HW tensor on the bottom/right so H and W are divisible by `multiple`."""
    h, w = tensor.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # F.pad pads last dim first: (left, right, top, bottom)
    return F.pad(tensor, (0, pad_w, 0, pad_h)), (h, w)  # also return original size to crop back


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1, # CHANGED from 2 to 1
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # --- get test pairs from metadata, same source of truth as training ---
    metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
    test_rows = metadata[metadata['split'] == 'test']
    image_paths = test_rows['tiff_image_path'].tolist()
    mask_paths = test_rows['tif_label_path'].tolist()

    print(f"found {len(image_paths)} test images.")
    if SHOW_PLOTS:
        print("close each plot window to see the next.")

    # running totals for dataset-level (micro-averaged) metrics
    total_tp = total_fp = total_fn = total_tn = 0
    per_image_records = []

    for img_rel, mask_rel in zip(image_paths, mask_paths):
        img_path = os.path.join(root, img_rel)
        mask_path = os.path.join(root, mask_rel)

        image = Image.open(img_path).convert("RGB")
        label = Image.open(mask_path)

        # preprocess exactly as training did, minus the random crop
        image_tensor = image_transform(image)                       # (3, H, W)
        label_array = (np.array(label) == 255).astype(np.int64)     # (H, W)

        # pad to multiple of 32 so encoder downsampling/upsampling stays aligned
        image_padded, orig_size = pad_to_multiple(image_tensor)
        input_batch = image_padded.unsqueeze(0).to(device)          # (1, 3, H', W')

        with torch.no_grad():
            output = model(input_batch)                             # (1, 1, H', W')
            probs = torch.sigmoid(output)                           # convert logits to 0-1 probabilities
            pred_mask = (probs > THRESHOLD).squeeze(1).long()       # drop channel dim -> (1, H', W')

        # crop back to the original (pre-pad) size before displaying
        h, w = orig_size
        pred_mask = pred_mask[:, :h, :w].squeeze(0).cpu().numpy()

        # --- per-image statistics ---
        tp, fp, fn, tn = compute_confusion_counts(pred_mask, label_array, positive_class=1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        img_metrics = metrics_from_counts(tp, fp, fn, tn)
        img_metrics_pct = {k: v * 100 for k, v in img_metrics.items()}
        per_image_records.append({
            "image": os.path.basename(img_rel),
            **img_metrics_pct,
        })

        print(
            f"{os.path.basename(img_rel):30s} "
            f"acc={img_metrics_pct['accuracy']:6.2f}%  "
            f"prec={img_metrics_pct['precision']:6.2f}%  "
            f"rec={img_metrics_pct['recall']:6.2f}%  "
            f"f1={img_metrics_pct['f1']:6.2f}%  "
            f"iou={img_metrics_pct['iou']:6.2f}%"
        )

        if SHOW_PLOTS:
            # un-normalize the image for display
            img_display = image_tensor.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = (img_display * std + mean).clip(0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_display)
            axes[0].set_title(os.path.basename(img_rel))
            axes[1].imshow(label_array, cmap='gray')
            axes[1].set_title("Ground Truth Mask")
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title(
                f"Predicted Mask\nIoU={img_metrics_pct['iou']:.1f}% "
                f"F1={img_metrics_pct['f1']:.1f}%"
            )
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()   # blocks until you close the window, then moves to next iteration

    # --- aggregate (dataset-level, micro-averaged) statistics ---
    overall = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)

    print("\n" + "=" * 60)
    print(f"Evaluated {len(image_paths)} test images")
    print("=" * 60)
    print(f"Accuracy : {overall['accuracy'] * 100:.2f}%")
    print(f"Precision: {overall['precision'] * 100:.2f}%")
    print(f"Recall   : {overall['recall'] * 100:.2f}%")
    print(f"F1 Score : {overall['f1'] * 100:.2f}%")
    print(f"IoU      : {overall['iou'] * 100:.2f}%")
    print("=" * 60)

    # also report the mean of per-image metrics (macro-average), useful to compare
    df = pd.DataFrame(per_image_records)
    print("\nPer-image macro-averaged metrics:")
    print(df[["accuracy", "precision", "recall", "f1", "iou"]].mean().round(2).to_string())

    # save full per-image breakdown to CSV next to the checkpoint
    out_csv = os.path.join(os.path.dirname(checkpoint_path), "test_metrics_per_image.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nPer-image metrics saved to: {out_csv}")


if __name__ == '__main__':
    main()