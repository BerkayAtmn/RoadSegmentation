# Road Segmentation — D-LinkNet34

A D-LinkNet34 (ResNet34 encoder + dilated-convolution center block + additive
LinkNet decoder) trained from scratch to segment roads in satellite/aerial
imagery, on a unified 512x512 corpus built from four public road datasets
(64,979 patches total). Includes a post-processing pipeline (denoise, close,
skeletonize) to turn the raw probability map into a clean centerline graph.

Built during a deep learning internship project — see [`notebook.ipynb`](notebook.ipynb)
for the full, ordered pipeline (train -> postprocess -> evaluate -> compare).

## Results

Evaluated on a held-out test split (3,175 patches) with relaxed
precision/recall/F1 on skeletons at a 3px tolerance (Mnih & Hinton metric):

| dataset   | patches | precision | recall | F1     |
|-----------|--------:|----------:|-------:|-------:|
| deepglobe |   2,137 |    0.6615 | 0.8591 | 0.7474 |
| mass      |     441 |    0.6928 | 0.8518 | 0.7641 |
| rngdet    |     432 |    0.6231 | 0.7756 | 0.6910 |
| spacenet  |     165 |    0.4563 | 0.5515 | 0.4994 |
| **ALL**   |   3,175 |    0.6433 | 0.8135 | 0.7185 |

Trained for 30 epochs (55,500 steps, batch 32) on an RTX 5060 8GB, reaching a
best validation balanced relaxed-F1 of 0.887 at ~144 img/s.

![predictions](results/predictions.png)
![postprocess](results/postprocess.png)

## Project layout

```
.
├── notebook.ipynb          # the project end-to-end: train, postprocess, evaluate, compare
├── src/
│   ├── dlinknet.py                  # D-LinkNet34 model definition
│   ├── road_dataset.py              # Dataset + augmentation + normalization
│   ├── prepare_unified_dataset.py   # builds unified_dataset/ from data/
│   ├── postprocess.py               # standalone: probability map -> clean mask + skeleton
│   └── visualize_predictions.py     # standalone: qualitative prediction grid
├── results/                 # sample output figures
├── requirements.txt
└── data/, unified_dataset/, trained_models/   # not tracked in git — see below
```

`notebook.ipynb` is the source of truth for the training run; the modules
under `src/` are the same code extracted so it can be reused as plain
scripts (e.g. `python src/visualize_predictions.py` after training).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the matching CUDA build of `torch`/`torchvision`
from [pytorch.org](https://pytorch.org/get-started/locally/) *before* running
`pip install -r requirements.txt` (this project was trained with `torch==2.13.0+cu132`).

The ResNet34 encoder is initialized from ImageNet weights pulled automatically
from the Hugging Face Hub (`smp-hub/resnet34.imagenet`) the first time training
runs — no manual download needed.

## Data

This repo does not include any imagery — `data/`, `unified_dataset/` and
`trained_models/` are git-ignored. The unified corpus merges four public road
datasets, normalized to 512x512 RGB PNGs + binary masks by
[`src/prepare_unified_dataset.py`](src/prepare_unified_dataset.py):

| dataset | expected path | source |
|---|---|---|
| DeepGlobe Road Extraction | `data/deepglobe/root/{train,val,test}/{images,masks}/` | [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467) (mirrored on [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)) |
| Massachusetts Roads | `data/mass/tiff/{train,val,test}/` + `{split}_labels/` | [Mass. Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) (Mnih), also mirrored on Kaggle |
| RNGDet / City-scale | `data/rngdet/{train,valid,test}/region_<id>_{sat,gt}.png` | the "city-scale" road network dataset used by the RNGDet / Sat2Graph papers (search for "RNGDet city-scale dataset" — download link is distributed via the paper authors' repo) |
| SpaceNet 3 (AOI 5 — Khartoum) | `data/AOI_5_Khartoum/{PS-RGB,geojson_roads}/` | [SpaceNet Roads Challenge, AOI 5](https://spacenet.ai/spacenet-roads-dataset/) |

SpaceNet ships road labels as GeoJSON centerlines rather than raster masks —
rasterize them into `data/AOI_5_Khartoum/masks/` yourself before running the
build (`prepare_unified_dataset.py` expects that folder to already exist).

Once `data/` is populated, build the unified corpus:

```bash
python src/prepare_unified_dataset.py            # full build (~30 GB output)
python src/prepare_unified_dataset.py --limit 8  # smoke test, 8 scenes per source/split
```

## Training

Open [`notebook.ipynb`](notebook.ipynb) and run the first cell (training),
or adapt it into a script — it expects `unified_dataset/` (built above) and
writes checkpoints to `trained_models/`. Config (epochs, batch size, LR,
loss weights) is at the top of the cell as plain constants.

The rest of the notebook — postprocessing, whole-split evaluation, and a
comparison against a prior U-Net baseline — reuses the same checkpoint.

## Notes

- `src/postprocess.py` and `src/visualize_predictions.py` are the same code
  as the notebook's postprocess/visualize cells, runnable standalone once a
  checkpoint exists at `trained_models/dlinknet34_best.pt`.
- Model design and postprocessing choices (why additive skips, why gaussian
  blur before thresholding, why no clDice loss, etc.) are documented as
  comments in [`src/dlinknet.py`](src/dlinknet.py), [`src/road_dataset.py`](src/road_dataset.py)
  and [`src/postprocess.py`](src/postprocess.py) — they explain *why*, not
  just *what*.
