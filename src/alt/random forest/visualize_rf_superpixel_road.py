"""
Visualize predictions from rf_superpixel_road_classifier.joblib on satellite images.

Mirrors the exact feature extraction used at training time (SLIC superpixels +
9 hand-crafted color/texture/shape features -> RandomForestClassifier), then
overlays predicted road superpixels on the original image.

Usage:
    python visualize_rf_superpixel_road.py --image path/to/tile.tif --model rf_superpixel_road_classifier.joblib
    python visualize_rf_superpixel_road.py --image path/to/tile.tif --model rf.joblib --out result.png --alpha 0.45
    python visualize_rf_superpixel_road.py --image_dir path/to/folder --model rf.joblib --out_dir results/
"""

import argparse
import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, mark_boundaries, slic

# --- Must match training config exactly (from the extraction script) ---
N_SEGMENTS = 10000
COMPACTNESS = 3.0
SIGMA = 0.5
GLCM_LEVELS = 16
MIN_SUPERPIXEL_SIZE = 20

FEATURE_NAMES = [
    "brightness", "hue", "r_minus_g", "gli", "egi", "tgi",
    "glcm_entropy", "glcm_mean", "aspect_ratio",
]

EPS = 1e-6


def extract_features_for_inference(image_array):
    """
    Same math as the training script's extract_features, minus the label pass.
    Returns:
        feats        : (n_sp, 9) float32 array, one row per superpixel
        superpixel   : (H, W) int array of superpixel ids
        keep_mask    : (n_sp,) bool array — which superpixel ids have valid features
    """
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

    img_f = image_array.astype(np.float64)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    gray_array = rgb2gray(image_array)
    hsv_array = rgb2hsv(img_f / 255.0)

    brightness = region_mean((r + g + b) / 3.0)
    hue        = region_mean(hsv_array[:, :, 0])
    r_minus_g  = region_mean(r - g)
    gli        = region_mean((2 * g - r - b) / (2 * g + r + b + EPS))
    egi        = region_mean(2 * g - r - b)
    tgi        = region_mean(g - 0.39 * r - 0.61 * b)
    glcm_mean  = region_mean(gray_array)

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
    hist += hist.transpose(0, 2, 1)

    totals = hist.sum(axis=(1, 2), keepdims=True)
    p = hist / np.maximum(totals, 1.0)
    logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    glcm_entropy = -(p * logp).sum(axis=(1, 2))

    major = np.zeros(n_sp, dtype=np.float64)
    minor = np.zeros(n_sp, dtype=np.float64)
    for prop in regionprops(superpixel + 1):
        i = prop.label - 1
        if 0 <= i < n_sp:
            # getattr fallback keeps this working across skimage versions
            # (major_axis_length/minor_axis_length were renamed to
            # axis_major_length/axis_minor_length; same values either way)
            major[i] = getattr(prop, "axis_major_length", None) or prop.major_axis_length
            minor[i] = getattr(prop, "axis_minor_length", None) or prop.minor_axis_length

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratio = np.where(minor > 0, major / np.maximum(minor, EPS), np.nan)

    feats = np.stack([
        brightness, hue, r_minus_g, gli, egi, tgi,
        glcm_entropy, glcm_mean, aspect_ratio,
    ], axis=1)

    keep_mask = (counts >= MIN_SUPERPIXEL_SIZE) & np.isfinite(feats).all(axis=1)

    return feats.astype(np.float32), superpixel, keep_mask


def predict_road_mask(image_array, clf, road_prob_threshold=0.5):
    """
    Runs SLIC + feature extraction + RF prediction, returns:
        road_mask   : (H, W) bool, True where pixel's superpixel is predicted road
        prob_mask   : (H, W) float, per-pixel road probability (from its superpixel)
        superpixel  : (H, W) int, superpixel id map (for boundary drawing)
    """
    feats, superpixel, keep_mask = extract_features_for_inference(image_array)
    n_sp = feats.shape[0]

    sp_pred = np.zeros(n_sp, dtype=np.int64)
    sp_prob = np.zeros(n_sp, dtype=np.float64)

    if keep_mask.any():
        valid_feats = feats[keep_mask]
        probs = clf.predict_proba(valid_feats)[:, 1]  # P(class == road)
        sp_prob[keep_mask] = probs
        sp_pred[keep_mask] = (probs >= road_prob_threshold).astype(np.int64)

    # Map per-superpixel predictions back onto the pixel grid
    road_mask = sp_pred[superpixel].astype(bool)
    prob_mask = sp_prob[superpixel]

    return road_mask, prob_mask, superpixel


def visualize(image_array, road_mask, prob_mask, superpixel, out_path,
              alpha=0.45, show_boundaries=True, show_prob_heatmap=True):
    """Saves a figure with: original | road overlay | probability heatmap."""
    n_panels = 3 if show_prob_heatmap else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    axes[0].imshow(image_array)
    axes[0].set_title("Original")
    axes[0].axis("off")

    overlay_base = image_array
    if show_boundaries:
        overlay_base = (mark_boundaries(image_array, superpixel, color=(1, 1, 0), mode="thin") * 255).astype(np.uint8)

    overlay = overlay_base.astype(np.float64).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay = np.where(
        road_mask[..., None],
        overlay * (1 - alpha) + red * alpha,
        overlay,
    ).astype(np.uint8)

    axes[1].imshow(overlay)
    axes[1].set_title(f"Predicted road (red) — {road_mask.mean():.1%} of area")
    axes[1].axis("off")

    if show_prob_heatmap:
        im = axes[2].imshow(prob_mask, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[2].set_title("Road probability")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_one(image_path, clf, out_path, alpha, threshold):
    image_array = np.array(Image.open(image_path).convert("RGB"))
    road_mask, prob_mask, superpixel = predict_road_mask(image_array, clf, road_prob_threshold=threshold)
    visualize(image_array, road_mask, prob_mask, superpixel, out_path, alpha=alpha)
    print(f"  {os.path.basename(image_path)}: {road_mask.mean():.2%} predicted road -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to rf_superpixel_road_classifier.joblib")
    parser.add_argument("--image", help="Path to a single satellite image")
    parser.add_argument("--image_dir", help="Directory of images to process (batch mode)")
    parser.add_argument("--out", default=None, help="Output image path (single-image mode). Default: <image>_predv2.png")
    parser.add_argument("--out_dir", default="rf_road_predictions", help="Output directory (batch mode)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity for predicted road (0-1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Road probability threshold")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image_dir")

    print(f"Loading classifier from {args.model} ...")
    clf = joblib.load(args.model)

    if args.image:
        out_path = args.out or (os.path.splitext(args.image)[0] + "_predv2.png")
        process_one(args.image, clf, out_path, args.alpha, args.threshold)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        paths = sorted(
            p for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
            for p in glob.glob(os.path.join(args.image_dir, ext))
        )
        if not paths:
            print(f"No images found in {args.image_dir}")
            return
        print(f"Found {len(paths)} images.")
        for p in paths:
            out_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0] + "_predv3.png")
            process_one(p, clf, out_path, args.alpha, args.threshold)


if __name__ == "__main__":
    main()