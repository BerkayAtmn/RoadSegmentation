import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode

import numpy as np
import pandas as pd
from PIL import Image

import os

root_massachusetts = r"C:\Dev\deeplearning\data\massachusetts-roads-dataset"
root_deepglobe = r"C:\Dev\deeplearning\data\deepglobe"

# Geometric ops only — cheap on uint8, and keeping the tensors uint8 here means
# 4x less data crossing worker -> pinned memory -> GPU. ToDtype/Normalize now
# happen on the GPU after transfer (see notebook).
train_transform = v2.Compose([
    v2.RandomCrop(size=(512, 512)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])

# No crop: validation runs sliding-window inference over the full tile,
# so it needs the tile at full resolution. Nothing else to do on CPU.
val_transform = None


def _read_image_uint8(path, mode):
    """torchvision's read_image doesn't decode TIFF (Massachusetts ships .tiff/.tif);
    fall back to PIL/libtiff for whatever it can't handle."""
    try:
        return read_image(path, mode=mode)
    except RuntimeError:
        pil_mode = 'RGB' if mode == ImageReadMode.RGB else 'L'
        arr = np.array(Image.open(path).convert(pil_mode))
        arr = arr[None, :, :] if arr.ndim == 2 else arr.transpose(2, 0, 1)
        return torch.from_numpy(arr.copy())


class RoadImageDataset(Dataset):
    # (image_col, mask_col) for each supported layout
    COLUMN_SETS = [
        ('tiff_image_path', 'tif_label_path'),   # Massachusetts (the png_* columns' files aren't on disk)
        ('sat_image_path', 'mask_path'),         # DeepGlobe
    ]

    def __init__(self, root, split='train', transform=None, val_fraction=0.0, seed=42):
        self.root = root
        self.split = split
        self.transform = transform

        metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))

        for img_col, msk_col in self.COLUMN_SETS:
            if img_col in metadata.columns and msk_col in metadata.columns:
                break
        else:
            raise ValueError(f"unrecognised metadata columns: {list(metadata.columns)}")

        # DeepGlobe ships masks only for the train split; valid/test rows are empty
        metadata = metadata.dropna(subset=[msk_col])

        if val_fraction > 0:
            # carve val out of train (needed for DeepGlobe)
            train_rows = metadata[metadata['split'] == 'train']
            shuffled = train_rows.sample(frac=1.0, random_state=seed)
            n_val = int(len(shuffled) * val_fraction)
            rows = shuffled.iloc[:n_val] if split == 'val' else shuffled.iloc[n_val:]
        else:
            rows = metadata[metadata['split'] == split]

        self.pairs = [
            (img, msk) for img, msk in zip(rows[img_col].tolist(), rows[msk_col].tolist())
            if os.path.exists(os.path.join(root, img)) and os.path.exists(os.path.join(root, msk))
        ]

        n_dropped = len(rows) - len(self.pairs)
        if n_dropped:
            print(f"[!] {root}: dropped {n_dropped}/{len(rows)} {split} rows with missing files")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image_path = os.path.join(self.root, image_path)
        mask_path = os.path.join(self.root, mask_path)

        # Read directly into PyTorch tensors (returns uint8)
        image_tensor = _read_image_uint8(image_path, ImageReadMode.RGB)
        mask_tensor = _read_image_uint8(mask_path, ImageReadMode.GRAY)

        # Fast tensor boolean logic
        mask_tv = tv_tensors.Mask((mask_tensor > 127).to(torch.uint8))
        image_tv = tv_tensors.Image(image_tensor)

        if self.transform is not None:
            image_tensor, label_tensor = self.transform(image_tv, mask_tv)
        else:
            # No CPU-side transform (e.g. val): pass the tile through unchanged, still
            # uint8. Dtype conversion/normalization happens on the GPU (see notebook).
            image_tensor, label_tensor = image_tv, mask_tv

        return image_tensor, label_tensor