import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image

from skimage.segmentation import slic
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import regionprops

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import joblib


root_massachusetts = r"D:\osm_data\Massachusetts"

# --- Extraction config ---
IMAGE_LIMIT = None
N_SEGMENTS = 1000
COMPACTNESS = 15.0
SIGMA = 0.5
GLCM_LEVELS = 16
MIN_SUPERPIXEL_SIZE = 20
N_WORKERS = max(1, mp.cpu_count() - 1)
ROAD_THRESHOLD = 0.1

FEATURE_NAMES = [
    "brightness", "hue", "r_minus_g", "gli", "egi", "tgi",
    "glcm_entropy", "glcm_mean", "aspect_ratio",
]

EPS = 1e-6


def extract_features(paths):
    """Extract per-superpixel features for one image. Returns (feats, labels) or None."""
    image_path, label_path = paths

    try:
        image_array = np.array(Image.open(image_path).convert("RGB"))
        label_array = np.array(Image.open(label_path).convert("L"))
    except Exception as exc:
        print(f"failed to read {image_path}: {exc}")
        return None

    label_array = (label_array > 127).astype(np.uint8)

    superpixel = slic(
        image_array,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0,
        channel_axis=-1,
    )

    n_sp = int(superpixel.max()) + 1
    labels_flat = superpixel.ravel()
    counts = np.bincount(labels_flat, minlength=n_sp).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)

    def region_mean(per_pixel_array):
        sums = np.bincount(
            labels_flat,
            weights=per_pixel_array.ravel().astype(np.float64),
            minlength=n_sp,
        )
        return sums / safe_counts

    # --- Color / spectral (one pass each, all superpixels at once) ---
    img_f = image_array.astype(np.float64)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    gray_array = rgb2gray(image_array)              # 0..1 float
    hsv_array = rgb2hsv(img_f / 255.0)

    brightness = region_mean((r + g + b) / 3.0)
    hue        = region_mean(hsv_array[:, :, 0])
    r_minus_g  = region_mean(r - g)
    gli        = region_mean((2 * g - r - b) / (2 * g + r + b + EPS))
    egi        = region_mean(2 * g - r - b)
    tgi        = region_mean(g - 0.39 * r - 0.61 * b)
    glcm_mean  = region_mean(gray_array)

    # --- GLCM entropy, distance=1, angle=0, symmetric, normed ---
    # Only pairs where both pixels belong to the same superpixel are counted,
    # so background pixels never pollute the texture statistics.
    L = GLCM_LEVELS
    q = np.clip((gray_array * (L - 1)).round(), 0, L - 1).astype(np.int64)

    qa = q[:, :-1].ravel()
    qb = q[:, 1:].ravel()
    la = superpixel[:, :-1].ravel()
    lb = superpixel[:, 1:].ravel()

    same = la == lb
    qa, qb, lab = qa[same], qb[same], la[same]

    idx = lab * (L * L) + qa * L + qb
    hist = np.bincount(idx, minlength=n_sp * L * L).reshape(n_sp, L, L).astype(np.float64)
    hist += hist.transpose(0, 2, 1)                                  # symmetric=True

    totals = hist.sum(axis=(1, 2), keepdims=True)
    p = hist / np.maximum(totals, 1.0)                               # normed=True
    logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    glcm_entropy = -(p * logp).sum(axis=(1, 2))

    # --- Shape: one regionprops call for the whole label image ---
    major = np.zeros(n_sp, dtype=np.float64)
    minor = np.zeros(n_sp, dtype=np.float64)
    for prop in regionprops(superpixel + 1):        # regionprops needs labels >= 1
        i = prop.label - 1
        if 0 <= i < n_sp:
            major[i] = prop.major_axis_length
            minor[i] = prop.minor_axis_length

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratio = np.where(minor > 0, major / np.maximum(minor, EPS), np.nan)

    # --- Labels ---
    road_fraction = region_mean(label_array.astype(np.float64))
    y = (road_fraction >= ROAD_THRESHOLD).astype(np.int64)

    feats = np.stack([
        brightness, hue, r_minus_g, gli, egi, tgi,
        glcm_entropy, glcm_mean, aspect_ratio,
    ], axis=1)

    keep = (counts >= MIN_SUPERPIXEL_SIZE) & np.isfinite(feats).all(axis=1)
    return feats[keep].astype(np.float32), y[keep]


if __name__ == "__main__":

    metadata = pd.read_csv(os.path.join(root_massachusetts, "metadata.csv"))
    metadata = metadata.dropna(subset=["tif_label_path"])
    rows = metadata[metadata["split"] == "train"]

    pairs = []
    for img, msk in zip(rows["tiff_image_path"], rows["tif_label_path"]):
        image_path = os.path.join(root_massachusetts, img)
        label_path = os.path.join(root_massachusetts, msk)
        if os.path.exists(image_path) and os.path.exists(label_path):
            pairs.append((image_path, label_path))

    if IMAGE_LIMIT is not None:
        pairs = pairs[:IMAGE_LIMIT]

    print(f"Extracting from {len(pairs)} images using {N_WORKERS} workers...")

    t0 = time.time()
    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, out in enumerate(pool.imap_unordered(extract_features, pairs), start=1):
            if out is not None:
                results.append(out)
            if i % 10 == 0 or i == len(pairs):
                elapsed = time.time() - t0
                print(f"  {i}/{len(pairs)} images | {elapsed:.1f}s | {elapsed / i:.2f}s per image")

    elapsed = time.time() - t0
    print(f"Extraction done in {elapsed:.1f}s ({elapsed / max(len(pairs), 1):.2f}s per image)")

    if not results:
        raise SystemExit("No features extracted.")

    X = np.concatenate([f for f, _ in results])
    y = np.concatenate([l for _, l in results])

    print(f"Total superpixels: {len(y)} | road: {y.sum()} ({y.mean():.2%}) | non-road: {(1 - y).sum()}")

    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit("One class is nearly empty — check the road_fraction >= ROAD_THRESHOLD threshold.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["non-road", "road"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("F1 (road):", f1_score(y_test, y_pred, zero_division=0))

    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print(importances)

    joblib.dump(clf, "rf_superpixel_road_classifier.joblib")