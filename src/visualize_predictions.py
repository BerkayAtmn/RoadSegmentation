"""Visualize D-LinkNet34 predictions on a few held-out test patches.

Loads the best checkpoint, runs inference on a handful of test images, and
saves a grid of (image | ground-truth | pred @ each threshold) rows, each
annotated with the source dataset and per-sample IoU.

sigmoid(logit) > 0.5 (THRESHOLDS[0] below) is the training-time operating
point, but the relaxed soft-F1 loss only rewards being within RHO=2px of a
road, not being confident about it -- faint/occluded strokes can end up with
p just under 0.5 and vanish at that cutoff. Lower thresholds trade precision
for recall to recover them; compare columns to see if that trade is worth it
for your use case, and pick where the road-shaped noise becomes disqualifying.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dlinknet import DLinkNet34
from road_dataset import RoadDataset, normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "trained_models" / "dlinknet34_best.pt"
ROOT = REPO_ROOT / "unified_dataset"
OUT = REPO_ROOT / "results" / "predictions.png"
N_SAMPLES = 3
SEED = 1
THRESHOLDS = (0.1, 0.01, 0.005)   # sigmoid(logit) cutoffs, high to low

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = DLinkNet34(weights=None).to(DEVICE, memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[+] loaded checkpoint @ step {ckpt['step']}, balanced rF1 {ckpt['score']:.4f}")

    test_set = RoadDataset(ROOT, "test", augment=False)
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(test_set)), N_SAMPLES)

    n_cols = 2 + len(THRESHOLDS)
    fig, axes = plt.subplots(N_SAMPLES, n_cols, figsize=(3 * n_cols, 3 * N_SAMPLES))
    with torch.inference_mode():
        for row, idx in enumerate(idxs):
            image, mask, sid = test_set[idx]        # image (H,W,3) u8, mask (H,W,1) u8
            source = test_set.source_names[sid]

            x = normalize(image.unsqueeze(0), DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                logits = model(x)
            prob = torch.sigmoid(logits.float()).squeeze().cpu()
            gt = mask.squeeze().bool()

            axes[row, 0].imshow(image.numpy())
            axes[row, 0].set_title(f"{source}  #{idx}", fontsize=9)
            axes[row, 1].imshow(gt.numpy(), cmap="gray")
            axes[row, 1].set_title("ground truth", fontsize=9)
            for col, t in enumerate(THRESHOLDS, start=2):
                pred = prob > t
                iou = (pred & gt).sum().float() / (pred | gt).sum().clamp(min=1).float()
                axes[row, col].imshow(pred.numpy(), cmap="gray")
                axes[row, col].set_title(f"p>{t}  IoU={iou:.3f}", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"[+] saved {OUT}")


if __name__ == "__main__":
    main()
