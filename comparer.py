"""comparer.py — one job: interactively browse the same predicted-vs-ground-truth
overlay pair tester.py plots (red prediction / cyan ground truth), live.

Left/Right : previous/next tile
Up/Down    : threshold -/+ 0.05
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from road_dataset import RoadImageDataset
from tester import (
    checkpoint_path,
    root_massachusetts,
    root_deepglobe,
    predict_tile,
    overlay_mask,
    PRED_COLOR,
    GT_COLOR,
)

# matplotlib binds left/right to its own back/forward view history by default,
# which both fires alongside our handler and does nothing useful here — free
# the keys up so there's no double-handling.
plt.rcParams["keymap.back"] = []
plt.rcParams["keymap.forward"] = []

# Which dataset to browse: "massachusetts" or "deepglobe".
# (DeepGlobe's own test split ships without masks — see tester.py — so this falls
# back to the same held-out val carve-out tester.py uses for it.)
DATASET = "deepglobe"  # "massachusetts" or "deepglobe"

THRESHOLD = 0.3
THRESHOLD_STEP = 0.05


def load_dataset(name):
    if name == "massachusetts":
        return RoadImageDataset(root_massachusetts, "test", transform=None)
    elif name == "deepglobe":
        return RoadImageDataset(root_deepglobe, "val", transform=None, val_fraction=0.1, seed=42)
    raise ValueError(f"unknown dataset: {name!r}")


class Comparer:
    def __init__(self, model, dataset, device, threshold=THRESHOLD):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.threshold = threshold
        self.idx = 0

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def on_key(self, event):
        if event.key == "right":
            self.idx = (self.idx + 1) % len(self.dataset)
        elif event.key == "left":
            self.idx = (self.idx - 1) % len(self.dataset)
        elif event.key == "up":
            self.threshold = round(min(1.0, self.threshold + THRESHOLD_STEP), 2)
        elif event.key == "down":
            self.threshold = round(max(0.0, self.threshold - THRESHOLD_STEP), 2)
        else:
            return
        self.draw()

    def draw(self):
        image_u8, mask_u8 = self.dataset[self.idx]
        image_name = os.path.basename(self.dataset.pairs[self.idx][0])

        with torch.no_grad():
            probs = predict_tile(self.model, image_u8, self.device)
        pred = (probs > self.threshold).cpu().numpy().astype(bool)
        gt = mask_u8.squeeze(0).numpy().astype(bool)

        img_disp = image_u8.permute(1, 2, 0).numpy().astype(np.float32) / 255.0
        pred_overlay = overlay_mask(img_disp, pred, PRED_COLOR)
        gt_overlay = overlay_mask(img_disp, gt, GT_COLOR)

        for ax in self.axes:
            ax.cla()
            ax.axis("off")
        self.axes[0].imshow(pred_overlay)
        self.axes[0].set_title(f"Predicted roads (red) — thr={self.threshold:.2f}")
        self.axes[1].imshow(gt_overlay)
        self.axes[1].set_title("Ground-truth roads (cyan)")
        self.fig.suptitle(
            f"[{self.idx + 1}/{len(self.dataset)}] {image_name}    "
            "(←/→ tile, ↑/↓ threshold)"
        )
        self.fig.canvas.draw_idle()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()

    dataset = load_dataset(DATASET)
    print(f"[+] {DATASET}: {len(dataset)} tiles. Left/Right = tile, Up/Down = threshold.")
    print("    (click the plot window once if the arrow keys don't respond)")

    comparer = Comparer(model, dataset, device)

    # The window needs OS keyboard focus to receive arrow keys at all — on TkAgg
    # it doesn't get that automatically, so force it once the window is up.
    get_tk_widget = getattr(comparer.fig.canvas, "get_tk_widget", None)
    if get_tk_widget is not None:
        try:
            tk_widget = get_tk_widget()
            tk_widget.after(200, tk_widget.focus_force)
        except Exception:
            pass

    plt.show()


if __name__ == "__main__":
    main()
