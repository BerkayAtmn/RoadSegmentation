"""Dataset for the unified 512x512 road corpus.

Tensors leave the workers as uint8 in (H, W, C) layout. NHWC on the host means
the pinned-to-device copy is a linear memcpy and `.permute(0, 3, 1, 2)` on the
GPU yields a channels_last view with no transpose kernel; returning CHW would
force a strided device-side permute on every batch.
"""

import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2 import functional as TF

SPLITS = ("train", "val", "test")


class RoadDataset(Dataset):
    def __init__(self, root, split="train", augment=False, datasets=None, min_road_frac=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        rows = pd.read_csv(os.path.join(root, "metadata.csv"))
        rows = rows[rows["split"] == split]
        if datasets is not None:
            rows = rows[rows["dataset"].isin(list(datasets))]
        if min_road_frac is not None:
            rows = rows[rows["road_frac"] >= min_road_frac]
        if rows.empty:
            raise ValueError(f"no patches left for split={split!r} after filtering")

        self.root = root
        self.augment = augment
        self.images = rows["image_path"].tolist()
        self.masks = rows["mask_path"].tolist()
        self.sources = rows["dataset"].tolist()
        self.source_names = sorted(set(self.sources))
        self.source_ids = torch.tensor([self.source_names.index(s) for s in self.sources])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.root, self.images[idx]), mode=ImageReadMode.RGB)
        mask = read_image(os.path.join(self.root, self.masks[idx]), mode=ImageReadMode.GRAY)
        mask = (mask > 127).to(torch.uint8)

        if self.augment:
            k = random.randrange(4)
            if k:
                image = torch.rot90(image, k, (1, 2))
                mask = torch.rot90(mask, k, (1, 2))
            if random.random() < 0.5:
                image = torch.flip(image, (2,))
                mask = torch.flip(mask, (2,))

            # One float pass, one clamp. Chaining the uint8 TF.adjust_* calls
            # clips highlights to 255 after brightness, before contrast sees them.
            if random.random() < 0.8:
                x = image.float()
                x = x * random.uniform(0.75, 1.25)
                m = x.mean()
                x = (x - m) * random.uniform(0.75, 1.25) + m
                g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
                x = (x - g) * random.uniform(0.7, 1.3) + g
                image = x.clamp_(0, 255).to(torch.uint8)
            if random.random() < 0.3:
                image = TF.adjust_hue(image, random.uniform(-0.04, 0.04))

        return (image.permute(1, 2, 0).contiguous(),
                mask.permute(1, 2, 0).contiguous(),
                self.source_ids[idx])

    def stratified_indices(self, power=0.5, generator=None):
        """Per-epoch order: each source gets its sqrt-balanced quota, drawn without
        replacement within the source and cycled if the quota exceeds its size.

        WeightedRandomSampler(replacement=True) draws each DeepGlobe patch
        59212/(430.5*sqrt(38957)) = 0.70 times per epoch, so exp(-0.70) = 50% of
        them are never seen. At EPOCHS=1 there is no second pass to recover them.
        """
        by_source = {}
        for i, s in enumerate(self.sources):
            by_source.setdefault(s, []).append(i)
        counts = {s: len(v) for s, v in by_source.items()}
        w = {s: n ** (1 - power) for s, n in counts.items()}
        scale = sum(counts.values()) / sum(w.values())

        out = []
        for s, idx in by_source.items():
            perm = torch.randperm(len(idx), generator=generator).tolist()
            for j in range(int(round(w[s] * scale))):
                out.append(idx[perm[j % len(idx)]])
        order = torch.randperm(len(out), generator=generator).tolist()
        return [out[i] for i in order]


_STATS = {}


def normalize(images, device):
    """uint8 (B, H, W, C) on the host -> normalised float32 channels_last on `device`."""
    key = (device.type, device.index)
    if key not in _STATS:
        _STATS[key] = (
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )
    mean, std = _STATS[key]
    images = images.to(device, non_blocking=True).permute(0, 3, 1, 2)
    return images.float().div_(255.0).sub_(mean).div_(std)