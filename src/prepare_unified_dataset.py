"""Merge DeepGlobe, Massachusetts Roads, RNGDet (City-scale) and SpaceNet 3
(AOI 5 Khartoum) into one unified 512x512 road-segmentation dataset.

Everything is normalised to 8-bit RGB PNG images + single-channel {0,255} PNG masks,
tiled with a sliding window, and indexed by a single metadata.csv so the training
dataloader never has to touch the source trees again.

SpaceNet ships roads as GeoJSON centrelines rather than raster masks; run
generate_spacenet_masks.py once first to rasterise them.

Run:
    python prepare_unified_dataset.py                 # full build
    python prepare_unified_dataset.py --limit 8       # smoke test (8 scenes/source-split)
    python prepare_unified_dataset.py --workers 8

    # add or rebuild one source without touching the others' 30 GB of patches
    python prepare_unified_dataset.py --datasets spacenet --append

Layout produced under OUT_ROOT:
    unified_dataset/{train,val,test}/{images,masks}/<dataset>_<id>_r<row_px>_c<col_px>.png
    unified_dataset/metadata.csv
    unified_dataset/dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # source scenes are up to 2048^2; PIL's bomb guard is irrelevant here

# ----------------------------------------------------------------------------- config
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUT_ROOT = Path(__file__).resolve().parent.parent / "unified_dataset"

PATCH = 512
TRAIN_STRIDE = 384          # 25% overlap -> more context/augmentation surface on train
EVAL_STRIDE = PATCH         # non-overlapping, so val/test patches are disjoint samples

# A patch is "positive" if it contains >=1 road pixel. Pure-background patches are
# mostly empty fields/water and swamp the loss, but dropping all of them removes the
# model's only evidence of what "no road" looks like -- so keep a seeded random few.
NEG_MAX_FRACTION = 0.05     # negatives may be at most 5% of the retained patches

# Massachusetts tiles carry large pure-white void regions (up to ~50% of a tile) where
# the aerial mosaic has no coverage, and the OSM-derived labels are still drawn across
# them. Those patches teach the model to hallucinate roads on blank canvas, so drop any
# patch that is mostly void. Measured ~0 on DeepGlobe/RNGDet, so this is a uniform rule
# that in practice only bites Massachusetts.
NODATA_MAX_FRACTION = 0.20

# level 6 measured identical output size to level 3 on these scenes (449.6 vs 450.6 KB)
# for several times the CPU, so 3 is the free lunch.
PNG_COMPRESS_LEVEL = 3

SEED = 42


@dataclass(frozen=True)
class Scene:
    """One source image/mask pair, plus where it lands in the unified dataset."""
    dataset: str
    split: str          # already normalised to train/val/test
    source_id: str
    image_path: Path
    mask_path: Path


# ----------------------------------------------------------------------- discovery
def discover_deepglobe(root: Path) -> list[Scene]:
    """<root>/root/{train,val,test}/{images/<id>_sat.jpg, masks/<id>_mask.png}.

    The 4982/622/622 split is a re-split of DeepGlobe's 6226 labelled train scenes
    (the official val/test have no public masks), and it is disjoint by scene id.
    """
    scenes = []
    for split in ("train", "val", "test"):
        img_dir, msk_dir = root / "root" / split / "images", root / "root" / split / "masks"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*_sat.jpg")):
            sid = img.name[: -len("_sat.jpg")]
            msk = msk_dir / f"{sid}_mask.png"
            if msk.exists():
                scenes.append(Scene("deepglobe", split, sid, img, msk))
    return scenes


def discover_mass(root: Path) -> list[Scene]:
    """<root>/tiff/{train,val,test}/<id>.tiff paired with <split>_labels/<id>.tif."""
    scenes = []
    for split in ("train", "val", "test"):
        img_dir, msk_dir = root / "tiff" / split, root / "tiff" / f"{split}_labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.tif*")):
            msk = msk_dir / f"{img.stem}.tif"
            if msk.exists():
                scenes.append(Scene("mass", split, img.stem, img, msk))
    return scenes


def discover_rngdet(root: Path) -> list[Scene]:
    """<root>/{train,valid,test}/region_<n>_{sat,gt}.png (split folders built from
    the dataset's own data_split.json)."""
    scenes = []
    for src_split, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        d = root / src_split
        if not d.is_dir():
            continue
        for img in sorted(d.glob("*_sat.png")):
            sid = img.name[: -len("_sat.png")]
            msk = d / f"{sid}_gt.png"
            if msk.exists():
                scenes.append(Scene("rngdet", split, sid, img, msk))
    return scenes


SPACENET_STEM = "SN3_roads_train_AOI_5_Khartoum"
SPACENET_SPLIT = (0.80, 0.10)      # train, val; remainder -> test


def discover_spacenet(root: Path) -> list[Scene]:
    """<root>/PS-RGB/<stem>_PS-RGB_img<n>.tif paired with the masks rasterised from
    GeoJSON by generate_spacenet_masks.py.

    SpaceNet only publishes labels for its train partition, so all 283 Khartoum tiles
    arrive unsplit. They get a seeded scene-level 80/10/10 split, mirroring how
    DeepGlobe's 6226 labelled scenes were re-split 4982/622/622 in this corpus. The
    split is by tile id, so no tile contributes patches to two splits.
    """
    img_dir, msk_dir = root / "PS-RGB", root / "masks"
    if not img_dir.is_dir():
        return []
    if not msk_dir.is_dir():
        print(f"[!] {msk_dir} not found -- run generate_spacenet_masks.py first; skipping spacenet")
        return []

    ids = sorted((p.name.split("_img")[-1][: -len(".tif")] for p in img_dir.glob("*.tif")),
                 key=lambda x: int(x))
    shuffled = list(ids)
    random.Random(f"{SEED}-spacenet-split").shuffle(shuffled)
    n_train = int(len(shuffled) * SPACENET_SPLIT[0])
    n_val = int(len(shuffled) * SPACENET_SPLIT[1])
    assignment = {}
    for i, sid in enumerate(shuffled):
        assignment[sid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    scenes = []
    for sid in ids:
        img = img_dir / f"{SPACENET_STEM}_PS-RGB_img{sid}.tif"
        msk = msk_dir / f"{SPACENET_STEM}_img{sid}.png"
        if msk.exists():
            scenes.append(Scene("spacenet", assignment[sid], f"img{sid}", img, msk))
    return scenes


def discover_all(data_root: Path, only: set[str] | None = None) -> list[Scene]:
    sources = (("deepglobe", discover_deepglobe, "deepglobe"),
               ("mass", discover_mass, "mass"),
               ("rngdet", discover_rngdet, "rngdet"),
               ("spacenet", discover_spacenet, "AOI_5_Khartoum"))
    scenes = []
    for name, fn, dirname in sources:
        if only and name not in only:
            continue
        d = data_root / dirname
        if not d.is_dir():
            print(f"[!] missing source directory {d} -- skipping {name}")
            continue
        found = fn(d)
        if not found:
            print(f"[!] {name}: discovered 0 scenes under {d}")
        scenes.extend(found)
    return scenes


# ------------------------------------------------------------------------ io + norm
def _read_raw(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    with Image.open(path) as im:
        if im.mode == "P":          # palette PNGs would otherwise decode to palette indices
            im = im.convert("RGB")
        return np.array(im)


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """-> (H, W, 3) uint8. 16-bit/float sources get a per-channel 2nd-98th percentile
    stretch, which is the standard remote-sensing display stretch; 8-bit sources are
    passed through untouched so we never re-quantise data that is already correct."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]         # drop alpha / extra bands
    if arr.shape[2] != 3:
        raise ValueError(f"cannot coerce image with {arr.shape[2]} channels to RGB")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    # Percentiles are taken over valid pixels only. SpaceNet tiles carry large pure-black
    # nodata wedges (one tile is 97.7% void); including those would drag the 2nd
    # percentile to 0 and waste most of the output range on empty space.
    valid = ~(arr == 0).all(axis=2)
    src = arr[valid] if valid.mean() > 0.02 else arr.reshape(-1, arr.shape[2])

    out = np.empty(arr.shape, dtype=np.uint8)
    for c in range(3):
        lo, hi = np.percentile(src[:, c].astype(np.float32), (2.0, 98.0))
        if hi <= lo:                # flat channel -> avoid divide-by-zero
            hi = lo + 1.0
        ch = arr[:, :, c].astype(np.float32)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    out[~valid] = 0                 # keep voids pure black so the nodata filter still sees them
    return out


def to_binary_mask(arr: np.ndarray) -> np.ndarray:
    """-> (H, W) uint8 in {0, 255}.

    max-over-channels (not luminance) so colour-coded masks survive: a pure-red road
    (255,0,0) has luminance 76 and would be thresholded away as background.
    """
    if arr.ndim == 3:
        arr = arr[:, :, :3].max(axis=2)

    if arr.dtype == np.uint8:
        binary = arr >= 128
    elif np.issubdtype(arr.dtype, np.floating):
        # float masks are either 0..1 or 0..255 depending on the exporter
        peak = float(arr.max())
        binary = arr >= (0.5 if peak <= 1.0 else 128.0)
    else:
        info = np.iinfo(arr.dtype)
        binary = arr.astype(np.float64) >= (info.max / 2.0)

    return np.where(binary, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- tiling
def plan_offsets(size: int, patch: int, stride: int, pad: bool) -> tuple[list[int], int]:
    """Sliding-window start positions along one axis, plus the size the axis must be
    padded to.

    pad=True  (val/test): ceil-cover with a non-overlapping grid, edge-pad the remainder.
    pad=False (train):    stay inside the image and clamp the final window flush to the
                          edge, so the border is covered by extra overlap rather than by
                          synthetic pixels.
    """
    if pad:
        n = max(1, math.ceil(size / patch))
        return [i * patch for i in range(n)], n * patch

    if size <= patch:
        return [0], patch           # smaller than a patch -> single padded window
    offsets = list(range(0, size - patch + 1, stride))
    if offsets[-1] + patch < size:
        offsets.append(size - patch)
    return offsets, size


def pad_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Edge-pad the bottom/right up to (h, w). Edge rather than reflect: reflecting a
    mask mirrors road geometry back into the scene and invents junctions."""
    ph, pw = h - arr.shape[0], w - arr.shape[1]
    if ph <= 0 and pw <= 0:
        return arr
    pad_spec = [(0, max(0, ph)), (0, max(0, pw))] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_spec, mode="edge")


def load_scene(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Normalised (image, mask) for one scene, guaranteed spatially aligned."""
    image = to_uint8_rgb(_read_raw(scene.image_path))
    mask = to_binary_mask(_read_raw(scene.mask_path))
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"{scene.dataset}/{scene.source_id}: image {image.shape[:2]} != mask {mask.shape[:2]}"
        )
    return image, mask


def tile_grid(scene: Scene, h: int, w: int) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """[(row_px, col_px, row_idx, col_idx)] plus the canvas size the scene must reach.

    Nothing is edge-padded any more, on any split. Padding a scene whose size is not a
    multiple of 512 fabricates pixels AND fabricates the mask over them: SpaceNet's
    1300px tiles were padded to 1536, so 236 of the last patch row's 512 rows were the
    final real row smeared downward, complete with replicated road labels (Massachusetts
    had the same defect at 36 rows). Clamping the last window flush to the edge instead
    covers the scene completely with real pixels; the price is that the final window
    overlaps its neighbour, which is harmless for evaluation and far cheaper than
    scoring a model against invented ground truth.
    """
    stride = TRAIN_STRIDE if scene.split == "train" else EVAL_STRIDE
    rows, ph = plan_offsets(h, PATCH, stride, pad=False)
    cols, pw = plan_offsets(w, PATCH, stride, pad=False)
    grid = [(r, c, ri, ci) for ri, r in enumerate(rows) for ci, c in enumerate(cols)]
    return grid, ph, pw


# ------------------------------------------------------------------- pass 1: scan
def scan_scene(scene: Scene) -> list[tuple]:
    """Per-patch statistics with nothing written to disk, so the negative-sampling
    budget can be computed over the whole corpus before we commit any bytes."""
    image, mask = load_scene(scene)
    h, w = mask.shape
    grid, ph, pw = tile_grid(scene, h, w)
    if (ph, pw) != (h, w):
        image, mask = pad_to(image, ph, pw), pad_to(mask, ph, pw)

    road = mask > 0
    void = np.logical_or((image == 255).all(axis=2), (image == 0).all(axis=2))

    out = []
    area = PATCH * PATCH
    for r, c, ri, ci in grid:
        road_px = int(road[r:r + PATCH, c:c + PATCH].sum())
        nodata = float(void[r:r + PATCH, c:c + PATCH].mean())
        out.append((scene.dataset, scene.split, scene.source_id, r, c, ri, ci,
                    road_px, road_px / area, nodata, h, w))
    return out


# ------------------------------------------------------------------- pass 2: write
def write_scene(args: tuple[Scene, list[tuple[int, int, int, int]]]) -> list[tuple]:
    """Encode only the patches selection kept for this scene."""
    scene, keep = args
    if not keep:
        return []

    image, mask = load_scene(scene)
    h, w = mask.shape
    _, ph, pw = tile_grid(scene, h, w)
    if (ph, pw) != (h, w):
        image, mask = pad_to(image, ph, pw), pad_to(mask, ph, pw)

    img_dir = OUT_ROOT / scene.split / "images"
    msk_dir = OUT_ROOT / scene.split / "masks"

    written = []
    for r, c, ri, ci in keep:
        # identical slices on both arrays -> pixel-perfect alignment by construction
        img_crop = image[r:r + PATCH, c:c + PATCH]
        msk_crop = mask[r:r + PATCH, c:c + PATCH]

        name = f"{scene.dataset}_{scene.source_id}_r{r}_c{c}.png"
        Image.fromarray(img_crop, mode="RGB").save(
            img_dir / name, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        Image.fromarray(msk_crop, mode="L").save(
            msk_dir / name, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=True)

        road_px = int((msk_crop > 0).sum())
        written.append((name, scene.dataset, scene.split, scene.source_id, r, c, ri, ci,
                        road_px, road_px / (PATCH * PATCH), h, w))
    return written


# ---------------------------------------------------------------- pass 3: validate
def validate_pair(args: tuple[str, str]) -> tuple[str, str] | None:
    """Header-check the image (mode+size proves (512,512,3) without decoding pixels)
    and fully decode the mask to prove its values are exactly {0,255}."""
    split, name = args
    img_p = OUT_ROOT / split / "images" / name
    msk_p = OUT_ROOT / split / "masks" / name
    try:
        with Image.open(img_p) as im:
            if im.mode != "RGB":
                return name, f"image mode {im.mode} != RGB"
            if im.size != (PATCH, PATCH):
                return name, f"image size {im.size} != ({PATCH}, {PATCH})"
        with Image.open(msk_p) as mm:
            if mm.mode != "L":
                return name, f"mask mode {mm.mode} != L"
            if mm.size != (PATCH, PATCH):
                return name, f"mask size {mm.size} != ({PATCH}, {PATCH})"
            vals = np.unique(np.array(mm))
        if not set(vals.tolist()) <= {0, 255}:
            return name, f"mask values {vals.tolist()[:8]} outside {{0, 255}}"
    except Exception as exc:                       # noqa: BLE001 - report, don't crash the pool
        return name, f"unreadable: {exc}"
    return None


# -------------------------------------------------------------------------- driver
def select_patches(records: list[tuple]) -> tuple[dict, dict]:
    """Keep every positive patch; add a seeded random sample of negatives so they are
    at most NEG_MAX_FRACTION of the retained set, budgeted per (dataset, split) so no
    single source dominates the negative pool. Void-heavy patches are dropped outright
    and are never eligible as negatives."""
    # run_pool returns scenes in completion order, which varies between runs; sort so the
    # seeded negative sample below is actually reproducible.
    records = sorted(records, key=lambda r: (r[0], r[1], str(r[2]), r[3], r[4]))
    by_group = defaultdict(lambda: {"pos": [], "neg": [], "void": 0})
    for rec in records:
        ds, split, road_px, nodata = rec[0], rec[1], rec[7], rec[9]
        g = by_group[(ds, split)]
        if nodata > NODATA_MAX_FRACTION:
            g["void"] += 1
        elif road_px > 0:
            g["pos"].append(rec)
        else:
            g["neg"].append(rec)

    keep_by_scene = defaultdict(list)
    stats = {}
    for (ds, split), g in sorted(by_group.items()):
        n_pos = len(g["pos"])
        # n_neg / (n_pos + n_neg) <= f   <=>   n_neg <= f/(1-f) * n_pos
        budget = int(round(NEG_MAX_FRACTION / (1.0 - NEG_MAX_FRACTION) * n_pos))
        chosen_neg = g["neg"]
        if len(chosen_neg) > budget:
            chosen_neg = random.Random(f"{SEED}-{ds}-{split}").sample(g["neg"], budget)

        for rec in g["pos"] + chosen_neg:
            keep_by_scene[(ds, split, rec[2])].append((rec[3], rec[4], rec[5], rec[6]))

        stats[(ds, split)] = {
            "candidates": n_pos + len(g["neg"]) + g["void"],
            "positive": n_pos,
            "negative_available": len(g["neg"]),
            "negative_kept": len(chosen_neg),
            "void_dropped": g["void"],
            "kept": n_pos + len(chosen_neg),
        }

    for key in keep_by_scene:
        keep_by_scene[key].sort()
    return keep_by_scene, stats


def _init_worker(out_root: str) -> None:
    """Windows spawns fresh interpreters for workers, so they re-import this module and
    would otherwise see the default OUT_ROOT rather than whatever --out selected."""
    global OUT_ROOT
    OUT_ROOT = Path(out_root)


def run_pool(fn, items, workers, desc, unit="scene", chunksize=1):
    """chunksize=1 + as_completed gives smooth progress on the heavy per-scene passes;
    a larger chunksize amortises IPC over the tens of thousands of tiny validation tasks."""
    kwargs = dict(max_workers=workers, initializer=_init_worker, initargs=(str(OUT_ROOT),))
    with ProcessPoolExecutor(**kwargs) as pool:
        if chunksize > 1:
            stream = pool.map(fn, items, chunksize=chunksize)
            return list(tqdm(stream, total=len(items), desc=desc, unit=unit, smoothing=0.05))
        futures = [pool.submit(fn, it) for it in items]
        return [f.result() for f in tqdm(as_completed(futures), total=len(futures),
                                         desc=desc, unit=unit, smoothing=0.05)]


def _previous_stats(out_root: Path, rebuilt: list[str]) -> dict:
    """Keep the selection stats of datasets this run did not touch."""
    f = out_root / "dataset_summary.json"
    if not f.exists():
        return {}
    try:
        prev = json.loads(f.read_text()).get("per_dataset_split", {})
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in prev.items() if k.split("/")[0] not in rebuilt}


def print_table(title, headers, rows, totals=None):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    if totals:
        widths = [max(w, len(str(totals[i]))) for i, w in enumerate(widths)]
    line = "  ".join("-" * w for w in widths)
    print(f"\n{title}")
    print(line)
    print("  ".join(str(h).ljust(w) if i == 0 else str(h).rjust(w)
                    for i, (h, w) in enumerate(zip(headers, widths))))
    print(line)
    for r in rows:
        print("  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w)
                        for i, (v, w) in enumerate(zip(r, widths))))
    if totals:
        print(line)
        print("  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w)
                        for i, (v, w) in enumerate(zip(totals, widths))))
    print(line)


def main() -> int:
    global OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap scenes per (dataset, split) for a fast smoke test")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--overwrite", action="store_true",
                    help="delete an existing output directory instead of refusing")
    ap.add_argument("--datasets", default="",
                    help="comma-separated subset to process, e.g. 'spacenet'")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing build: keep other datasets' patches and "
                         "metadata rows, replacing only the datasets processed this run")
    args = ap.parse_args()
    selected = {d.strip() for d in args.datasets.split(",") if d.strip()} or None
    OUT_ROOT = args.out.resolve()

    t0 = time.time()
    print(f"sources : {DATA_ROOT}")
    print(f"output  : {OUT_ROOT}")
    print(f"workers : {args.workers}   patch={PATCH} train_stride={TRAIN_STRIDE} "
          f"eval_stride={EVAL_STRIDE}")

    scenes = discover_all(DATA_ROOT, only=selected)
    if not scenes:
        print("[x] no scenes discovered -- check DATA_ROOT", file=sys.stderr)
        return 1

    if args.limit:
        capped, seen = [], defaultdict(int)
        for s in scenes:
            if seen[(s.dataset, s.split)] < args.limit:
                capped.append(s)
                seen[(s.dataset, s.split)] += 1
        scenes = capped

    counts = defaultdict(int)
    for s in scenes:
        counts[(s.dataset, s.split)] += 1
    print_table("Discovered source scenes",
                ["dataset", "train", "val", "test", "total"],
                [[ds] + [counts[(ds, sp)] for sp in ("train", "val", "test")]
                 + [sum(counts[(ds, sp)] for sp in ("train", "val", "test"))]
                 for ds in sorted({d for d, _ in counts})])

    # leakage guard: a source id must not appear in two splits of the same dataset
    ids = defaultdict(set)
    for s in scenes:
        ids[(s.dataset, s.split)].add(s.source_id)
    for ds in sorted({d for d, _ in ids}):
        sp = [ids[(ds, x)] for x in ("train", "val", "test")]
        overlap = (sp[0] & sp[1]) | (sp[0] & sp[2]) | (sp[1] & sp[2])
        if overlap:
            print(f"[x] {ds}: {len(overlap)} scene ids appear in more than one split "
                  f"(e.g. {sorted(overlap)[:5]}) -- aborting", file=sys.stderr)
            return 1
    print("\n[ok] no scene-id overlap between splits in any dataset")

    built = sorted({s.dataset for s in scenes})
    if args.append:
        # Rebuilding one source is safe in isolation: the negative-sampling budget is
        # computed per (dataset, split) and the seeds are fixed, so the other datasets'
        # selections are untouched by definition. Clear only this run's own patches so a
        # re-run with different settings cannot leave orphans behind.
        removed = 0
        for split in ("train", "val", "test"):
            for kind in ("images", "masks"):
                d = OUT_ROOT / split / kind
                if not d.is_dir():
                    continue
                for f in d.glob("*.png"):
                    if f.name.split("_")[0] in built:
                        f.unlink(); removed += 1
        print(f"\nappend mode: rebuilding {built}; cleared {removed:,} existing files for them")
    elif OUT_ROOT.exists():
        if not args.overwrite:
            print(f"[x] {OUT_ROOT} already exists; pass --overwrite to replace it "
                  f"(or --append to add a dataset)", file=sys.stderr)
            return 1
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "masks").mkdir(parents=True, exist_ok=True)

    # -- pass 1 -------------------------------------------------------------
    t = time.time()
    scanned = run_pool(scan_scene, scenes, args.workers, "scan   ")
    records = [r for sub in scanned for r in sub]
    print(f"       {len(records):,} candidate patches in {time.time() - t:.1f}s")

    keep_by_scene, stats = select_patches(records)
    n_keep = sum(len(v) for v in keep_by_scene.values())
    print(f"       {n_keep:,} patches selected for writing")

    # -- pass 2 -------------------------------------------------------------
    t = time.time()
    tasks = [(s, keep_by_scene.get((s.dataset, s.split, s.source_id), [])) for s in scenes]
    tasks = [x for x in tasks if x[1]]
    written = [r for sub in run_pool(write_scene, tasks, args.workers, "write  ") for r in sub]
    print(f"       wrote {len(written):,} image/mask pairs in {time.time() - t:.1f}s")

    # -- metadata -----------------------------------------------------------
    df = pd.DataFrame(written, columns=[
        "filename", "dataset", "split", "source_id", "row_px", "col_px",
        "row_idx", "col_idx", "road_pixels", "road_frac", "src_height", "src_width"])
    df["image_path"] = df["split"] + "/images/" + df["filename"]
    df["mask_path"] = df["split"] + "/masks/" + df["filename"]
    df["is_positive"] = df["road_pixels"] > 0
    df = df.sort_values(["dataset", "split", "source_id", "row_px", "col_px"]).reset_index(drop=True)
    df = df[["filename", "dataset", "split", "source_id", "image_path", "mask_path",
             "row_px", "col_px", "row_idx", "col_idx", "road_pixels", "road_frac",
             "is_positive", "src_height", "src_width"]]

    meta_path = OUT_ROOT / "metadata.csv"
    if args.append and meta_path.exists():
        prev = pd.read_csv(meta_path)
        kept = prev[~prev["dataset"].isin(built)]
        print(f"       merging {len(df):,} new rows into {len(kept):,} retained rows")
        df = pd.concat([kept, df], ignore_index=True)
    df = df.sort_values(["dataset", "split", "source_id", "row_px", "col_px"]).reset_index(drop=True)
    df.to_csv(meta_path, index=False)
    print(f"       metadata.csv: {len(df):,} rows x {len(df.columns)} cols")

    # -- pass 3 -------------------------------------------------------------
    t = time.time()
    problems = []
    for split in ("train", "val", "test"):
        img_names = {p.name for p in (OUT_ROOT / split / "images").glob("*.png")}
        msk_names = {p.name for p in (OUT_ROOT / split / "masks").glob("*.png")}
        for miss in sorted(img_names - msk_names)[:5]:
            problems.append((miss, f"{split}: image has no matching mask"))
        for miss in sorted(msk_names - img_names)[:5]:
            problems.append((miss, f"{split}: mask has no matching image"))
        expected = set(df.loc[df["split"] == split, "filename"])
        if img_names != expected:
            problems.append((split, f"on-disk images ({len(img_names)}) disagree with "
                                    f"metadata rows ({len(expected)})"))

    checks = list(zip(df["split"].tolist(), df["filename"].tolist()))
    for res in run_pool(validate_pair, checks, args.workers, "validate",
                        unit="pair", chunksize=64):
        if res:
            problems.append(res)
    print(f"       validated {len(checks):,} pairs in {time.time() - t:.1f}s")

    # -- summary ------------------------------------------------------------
    rows, tot = [], defaultdict(int)
    for (ds, split), s in sorted(stats.items()):
        rows.append([f"{ds}/{split}", s["candidates"], s["void_dropped"],
                     s["positive"], s["negative_available"], s["negative_kept"], s["kept"],
                     f"{100 * s['negative_kept'] / max(1, s['kept']):.1f}%"])
        for k in ("candidates", "void_dropped", "positive", "negative_available",
                  "negative_kept", "kept"):
            tot[k] += s[k]
    print_table(
        "Patch selection",
        ["dataset/split", "candidates", "void_drop", "positive", "neg_avail", "neg_kept",
         "kept", "neg%"],
        rows,
        ["TOTAL", tot["candidates"], tot["void_dropped"], tot["positive"],
         tot["negative_available"], tot["negative_kept"], tot["kept"],
         f"{100 * tot['negative_kept'] / max(1, tot['kept']):.1f}%"])

    pivot = df.pivot_table(index="dataset", columns="split", values="filename",
                           aggfunc="count", fill_value=0)
    for sp in ("train", "val", "test"):
        if sp not in pivot.columns:
            pivot[sp] = 0
    print_table("Final split distribution per source dataset",
                ["dataset", "train", "val", "test", "total", "mean road frac"],
                [[ds, int(pivot.loc[ds, "train"]), int(pivot.loc[ds, "val"]),
                  int(pivot.loc[ds, "test"]), int(pivot.loc[ds].sum()),
                  f"{df.loc[df.dataset == ds, 'road_frac'].mean():.4f}"]
                 for ds in pivot.index],
                ["TOTAL", int(pivot["train"].sum()), int(pivot["val"].sum()),
                 int(pivot["test"].sum()), len(df), f"{df['road_frac'].mean():.4f}"])

    size_gb = sum(p.stat().st_size for p in OUT_ROOT.rglob("*.png")) / 1e9
    print(f"\non-disk size: {size_gb:.2f} GB")

    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"patch": PATCH, "train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE,
                   "neg_max_fraction": NEG_MAX_FRACTION, "nodata_max_fraction": NODATA_MAX_FRACTION,
                   "png_compress_level": PNG_COMPRESS_LEVEL, "seed": SEED},
        "totals": {"patches": len(df),
                   "positive": int(df["is_positive"].sum()),
                   "negative": int((~df["is_positive"]).sum()),
                   "size_gb": round(size_gb, 3)},
        "per_dataset_split": {**(_previous_stats(OUT_ROOT, built) if args.append else {}),
                              **{f"{ds}/{sp}": v for (ds, sp), v in stats.items()}},
        "validation_problems": [list(p) for p in problems[:50]],
    }
    (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    if problems:
        print(f"\n[x] VALIDATION FAILED: {len(problems)} problem(s)")
        for name, why in problems[:20]:
            print(f"    {name}: {why}")
        return 1

    print(f"\n[ok] validation passed: every patch is a paired RGB {PATCH}x{PATCH} image "
          f"+ binary {{0,255}} mask")
    print(f"[ok] done in {time.time() - t0:.1f}s -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    sys.exit(main())
