# Running and fine-tuning LoRA adapter (Real-ESRGAN RRDB)

This document explains step-by-step how to prepare an environment, install dependencies, and run `train_lora_adapter.py` on Windows (PowerShell). It includes a recommended Conda flow (simpler on Windows), a pip/venv fallback, one-time setup, subsequent runs, and troubleshooting tips.

## Quick summary
- Recommended: use Anaconda/Miniconda and create a conda env named `lora`.
- If you must use a venv, install PyTorch first (special wheel for CUDA 12.8) before other packages.
- Run training with: `python .\train_lora_adapter.py --config .\configs\train_lora.yaml --mode small` (small = quick test).

---

## Checklist (what you'll do)
- [ ] Create/activate environment (Conda recommended).
- [ ] Install PyTorch (CUDA 12.8) and other packages.
- [ ] Verify GPU and torch.
- [ ] Run small test training and iterate.

---

## Recommended: Conda (best on Windows)
Use the Anaconda Prompt (or a PowerShell with conda initialized). This avoids building heavy packages from source.

Commands (run one-by-one in Anaconda Prompt):

```powershell
# create and activate env
conda create -n lora python=3.9 -y
conda activate lora

# install PyTorch for CUDA 12.8 using conda/pytorch channels (if available)
# if package not found on channels, see the pip instructions below
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.8 -y

# install common dependencies (conda-forge provides binaries)
conda install -c conda-forge pandas opencv pyyaml tqdm scikit-image matplotlib -y

# optional pip-only packages
pip install lpips realesrgan ultralytics torchmetrics
```

Verification:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import pandas, numpy, cv2, yaml; print(pandas.__version__, numpy.__version__, cv2.__version__)"
```

Notes:
- If `pytorch-cuda=12.8` is not available via conda channels for your conda version/platform, install PyTorch via pip (see venv/pip section below) after activating the `lora` env.
- You only need to install packages once. Later sessions only require `conda activate lora`.

---

## venv / pip-only fallback (if you prefer virtualenv)
PowerShell commands below. This flow assumes you have GPU with CUDA 12.8 (your `nvidia-smi` shows CUDA 12.8). If you want CPU-only, install CPU torch wheel instead.

```powershell
# create venv (one-time)
python -m venv .\venv_lora
# activate
.\venv_lora\Scripts\Activate.ps1

# upgrade pip and helpers
python -m pip install --upgrade pip setuptools wheel certifi

# IMPORTANT: install PyTorch first (special wheel). Use the official command from https://pytorch.org/get-started/locally/.
# Example (CUDA 12.8). Replace with exact command from pytorch.org if needed:
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# then install other dependencies (binary wheels)
python -m pip install numpy pandas opencv-python pyyaml tqdm scikit-image matplotlib

# install basicsr & realesrgan (try pip; if it fails, install from git)
python -m pip install basicsr realesrgan || \
python -m pip install git+https://github.com/xinntao/BasicSR.git@v1.4.2 git+https://github.com/xinntao/Real-ESRGAN.git@v0.3.0

# remaining packages
python -m pip install lpips ultralytics torchmetrics
```

Notes and common fixes:
- If pip attempts to build large packages and fails (cmake/ninja errors), prefer using conda or installing prebuilt wheels for numpy/pandas first.
- If you get SSL certificate errors, set the certifi path for the session before pip commands:

```powershell
$env:SSL_CERT_FILE = python -c "import certifi; print(certifi.where())"
```

---

## One-time vs subsequent runs
- One-time (first computer setup): follow the full install above (Conda recommended). Create the env and install packages.
- Subsequent runs on the same machine: just activate the env and run the script.

Commands (subsequent runs):

Conda:
```powershell
conda activate lora
python .\train_lora_adapter.py --config .\configs\train_lora.yaml --mode small
```

venv:
```powershell
.\venv_lora\Scripts\Activate.ps1
python .\train_lora_adapter.py --config .\configs\train_lora.yaml --mode small
```

---

## Running full training
In `train_lora_adapter.py`, `--mode small` sets small/quick defaults. To run full training:

```powershell
python .\train_lora_adapter.py --config .\configs\train_lora.yaml --mode full
```

The script reads `pairs_csv` and `model_path` from the YAML config. The code includes a fallback resolver:
- If the path in YAML (e.g. `D:\degraded_full_dataset\pairs.csv`) exists, it uses it.
- Otherwise it will try replacing the drive letter with `P:` and use that if it exists.

Per-row CSV columns are also resolved with a fallback: the dataset loader prefers `degraded_path` (D:) and falls back to `degraded_path(P)` (P:) when necessary.

---

## Running on another computer
- Recommended: export and reuse the environment instead of reinstalling from scratch.

Conda export (do this on the original machine once):
```powershell
conda activate lora
conda env export --no-builds > lora_env.yaml
```

On the other computer:
```powershell
conda env create -f lora_env.yaml
conda activate lora
```

If you used venv + pip, create a `requirements.txt` after setup:
```powershell
pip freeze > requirements_for_lora.txt
# on another machine (after creating same Python venv):
pip install -r requirements_for_lora.txt
```

Warning: PyTorch wheels are platform/CUDA dependent. On the other computer, you may need to install the correct `torch` wheel for the local CUDA version before running `pip install -r requirements...`.

---

## requirements_lora.txt advice
- Because `torch` and `torchvision` often require special installation (index or conda), it's best to remove them from `requirements_lora.txt` or keep a second file without torch.
- Suggested workflow:
  1. Install `torch`/`torchvision` first (conda or pip with special index).
  2. Then run `pip install -r requirements_lora_no_torch.txt` which excludes torch and torchvision.

---

## Troubleshooting
- ModuleNotFoundError: No module named 'yaml'
  - `pip install pyyaml` (or ensure `pyyaml` installed in the active env).

- basicsr build fails (error when pip builds wheels): make sure `torch` is installed first. If pip still fails, install BasicSR from Git:

```powershell
python -m pip install git+https://github.com/xinntao/BasicSR.git
```

- SSL certificate errors during pip builds:

```powershell
$env:SSL_CERT_FILE = python -c "import certifi; print(certifi.where())"
# then run pip commands again
```

- OpenMP conflict (libiomp5md.dll) error when importing torch:
  - Temporary test workaround (unsafe):

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'; python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

  - Permanent fix: prefer a single install source (conda) or find/remove duplicate `libiomp5md.dll` copies. To search:

```powershell
Get-ChildItem -Path $env:USERPROFILE -Filter libiomp5md.dll -Recurse -ErrorAction SilentlyContinue
```

---

## Example quick test (end-to-end)
1. Activate env (conda example):
```powershell
conda activate lora
```
2. Run quick training (2 epochs by `--mode small`):
```powershell
python .\train_lora_adapter.py --config .\configs\train_lora.yaml --mode small
```
3. If you see errors, copy the traceback and open an issue or ask for help with that specific error.

---

## Final notes
- After the first-time setup you only need to activate the environment for subsequent sessions.
- For portability between machines, export the conda env or provide a `requirements` file and a note about installing the correct PyTorch wheel for the target machine's CUDA.

If you want, I can:
- Create `requirements_lora_no_torch.txt` (a copy of `requirements_lora.txt` without torch/torchvision) in the repo.
- Generate a `conda` env YAML from your current env once it's fully installed.

---

End of guide.
