import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode

import numpy as np
import pandas as pd

import os

root_massachusetts = r"C:\Dev\deeplearning\data\massachusetts-roads-dataset"
root_deepglobe = r"C:\Dev\deeplearning\data\deepglobe"

# Define v2 transformation pipelines
train_transform = v2.Compose([
    v2.RandomCrop(size=(512, 512)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = v2.Compose([
    v2.CenterCrop(size=(512, 512)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RoadImageDataset(Dataset):
    # (image_col, mask_col) for each supported layout
    COLUMN_SETS = [
        ('png_image_path', 'png_label_path'),    # Massachusetts (tiff_/tif_ don't decode via read_image)
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
        image_tensor = read_image(image_path, mode=ImageReadMode.RGB)
        mask_tensor = read_image(mask_path, mode=ImageReadMode.GRAY)

        # Fast tensor boolean logic
        mask_tv = tv_tensors.Mask((mask_tensor > 127).to(torch.uint8))
        image_tv = tv_tensors.Image(image_tensor)

        if self.transform is not None:
            image_tensor, label_tensor = self.transform(image_tv, mask_tv)
        else:
            image_tensor = v2.functional.to_dtype(image_tv, torch.float32, scale=True)
            label_tensor = mask_tv

        return image_tensor, label_tensor