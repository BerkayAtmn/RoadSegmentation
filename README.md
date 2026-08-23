# Satellite Image Road Segmentation

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/BerkayAtmn/deeplearning
cd deeplearning
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Mac/Linux, activate with `source .venv/bin/activate` instead.

### 2. Install PyTorch (GPU build)

Install PyTorch **first and separately** — the correct command depends on your CUDA version.
Check yours with `nvidia-smi` (top-right of the output), then get the matching command from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

Do not just `pip install torch` from requirements.txt — that risks silently installing a
CPU-only build.

Verify it worked:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Should print `True` for CUDA availability.

### 3. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set up Kaggle API access

1. Create a Kaggle account if you don't have one: https://www.kaggle.com/account/login
2. Authenticate the CLI:
   ```powershell
   kaggle auth login
   ```
   (Opens a browser OAuth flow. Alternatively, generate a token at
   kaggle.com/settings/api and set it as an environment variable —
   see Kaggle CLI docs for your platform's syntax.)

### 5. Download the dataset

```powershell
kaggle datasets download -d balraj98/massachusetts-roads-dataset
```

Unzip the downloaded file into a `data/` folder in the project root. Expected structure:

```
data/
  tiff/
    train/
    train_labels/
    val/
    val_labels/
    test/
    test_labels/
```