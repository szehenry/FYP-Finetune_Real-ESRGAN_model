# FYP: LoRA Fine-Tuning of Real-ESRGAN for Drone Image Restoration

This repository implements **parameter-efficient fine-tuning** of **Real-ESRGAN (RRDBNet)** using **LoRA (Low-Rank Adaptation)** on convolutional layers. The goal is to improve restoration of **aerial / drone imagery** affected by **motion blur** (global and object-level) and **low-light** degradations, compared with the off-the-shelf pretrained model and classical baselines.

---

## High-level pipeline

1. **Dataset split** — Group images by source identity, then split into train / val / test (`split_dataset.py`).
2. **Degradation synthesis** — Generate synthetic degraded images and build pairs for supervised training (`degradation_synthesis.py`).
3. **LoRA training** — Freeze the Real-ESRGAN backbone; train only LoRA adapters with L1 + optional LPIPS loss (`train_lora_adapter.py` + YAML configs).
4. **Baselines** — Classical image-processing baselines with full-reference metrics (`baseline_evaluation_with_gt.py`; depends on `baseline_evaluation.py`).
5. **Evaluation** — pretrained **Real-ESRGAN only** (`evaluate_realesrgan_only_v2.py`) vs **LoRA fine-tuned** model (`evaluate_lora_model.py`) on the same `pairs.csv` and metric suite.

---

## Core scripts

| Script | Role |
|--------|------|
| `split_dataset.py` | Per-dataset grouped random split (train/val/test) with fixed seed; writes `train_list.txt`, `val_list.txt`, `test_list.txt`. |
| `degradation_synthesis.py` | Synthesizes degradations: global motion blur, object motion blur (YOLOv8-seg, optional), low-light; outputs degraded images and pairing metadata. |
| `train_lora_adapter.py` | LoRA training on paired LR/HR crops; checkpointing and resume; CUDA / MPS / CPU. |
| `configs/train_lora_mps.yaml` | Training config tuned for **Apple Silicon (MPS)** (e.g. `amp: false`). |
| `configs/train_lora_rtx4060.yaml` | Training config tuned for **NVIDIA RTX 4060** (e.g. `amp: true`). |
| `baseline_evaluation_with_gt.py` | Runs classical baselines + PSNR / SSIM / LPIPS and extra descriptors; uses `pairs.csv`. |
| `evaluate_realesrgan_only_v2.py` | Evaluates **stock Real-ESRGAN** (`RealESRGAN_x4plus`) with tiling, optional checkpoint resume, leaderboards per degradation type. |
| `evaluate_lora_model.py` | Same evaluation protocol for the **LoRA-wrapped** model; CLI for checkpoint, rank, alpha, paths. |

---

## Dependencies

Install Python 3.10+ (recommended). Typical packages (adjust per your environment):

- **Training / LoRA:** `torch`, `torchvision`, `pyyaml`, `opencv-python`, `tqdm`, `pandas`, `lpips` (for perceptual loss)
- **Real-ESRGAN eval:** `realesrgan`, `basicsr`
- **Degradation:** `numpy`, `opencv-python`, `torch`; optional `ultralytics` for object motion blur
- **Metrics:** `scikit-image`, `lpips`; optional `pyiqa` for NIQE/BRISQUE in some code paths

Project-specific requirement files (examples):

- `requirements_lora.txt`
- `requirements_degradation.txt`
- `requirements_baseline.txt`
- `requirements_realesrgan_baseline.txt`

---

## Configuration

### Training YAML (`configs/train_lora_mps.yaml` / `train_lora_rtx4060.yaml`)

Both configs share the same **logical** hyperparameters (example):

- `model_path`, `pairs_csv` — paths to **RealESRGAN_x4plus.pth** and **pairs.csv**
- `rank`, `alpha` — LoRA capacity and scaling (e.g. rank **32**, alpha **1.0**)
- `lr`, `batch_size`, `epochs`, `patch_size`, `scale` (typically **4×** SR)
- `use_perceptual_loss`, `perceptual_weight` — LPIPS term
- `use_scheduler`, `lr_milestones`, `lr_gamma`
- `checkpoint`, `epoched_checkpoints`, `save_lora_only`, `keep_last`, `save_every_n_iterations`

**Differences:**

| | `train_lora_mps.yaml` | `train_lora_rtx4060.yaml` |
|---|------------------------|----------------------------|
| **AMP** | `false` (MPS does not use CUDA AMP) | `true` (CUDA mixed precision) |
| **Paths in repo** | Example: `D:/...` | Example: `P:/...` |

Edit paths to match your machine. `train_lora_adapter.py` can **resolve** common roots (`D:/`, `P:/`, `/Volumes/Extreme SSD/`) for `model_path` and `pairs_csv` when files are missing at the original string.

### Other scripts (hard-coded paths)

`split_dataset.py`, `baseline_evaluation_with_gt.py`, and `evaluate_realesrgan_only_v2.py` contain **absolute paths** in their config classes. **Update these paths** before running on your system, or mirror the expected folder layout.

---

## Usage

### 1. Dataset split

```bash
python split_dataset.py
```

Edit `FYP_DIR`, `IMAGES_BASE_DIR`, `DATASET_FOLDERS`, `OUTPUT_DIR`, and split ratios at the top of the script. Outputs include `train_list.txt`, `val_list.txt`, `test_list.txt`, and `split_statistics.json`.

---

### 2. Degradation synthesis

```bash
python degradation_synthesis.py --split_dir <path_to_split_dir> --output_dir <degraded_output_dir>
```

Optional flags:

- `--device auto|cuda|cpu`
- `--max_images N` — limit images per split (debug)
- `--no_global_blur`, `--no_object_blur`, `--no_low_light`
- `--read_timeout` — skip slow/unavailable files (e.g. cloud sync)

Requires a split directory containing lists such as `train_list.txt` (as produced by your pipeline).

---

### 3. LoRA training

```bash
# Quick debug run (2 epochs, small patches — overrides config)
python train_lora_adapter.py --config configs/train_lora_mps.yaml --mode small

# Full training (uses epochs/patch_size from YAML)
python train_lora_adapter.py --config configs/train_lora_mps.yaml --mode full

# Resume from checkpoint
python train_lora_adapter.py --config configs/train_lora_mps.yaml --mode full --resume

# Custom checkpoint path (also enables resume)
python train_lora_adapter.py --config configs/train_lora_mps.yaml --mode full --checkpoint ./lora_checkpoint_r32.pth
```

**Prerequisites:** `pairs.csv` with columns including paired **degraded** and **clean** paths and `split` (e.g. `train/val`), and pretrained **RealESRGAN_x4plus.pth**.

---

### 4. Classical baselines (with ground truth)

```bash
python baseline_evaluation_with_gt.py
```

Requires `baseline_evaluation.py` in the same directory. Ensures `ConfigWithGT.PAIRS_CSV`, `DEGRADED_DIR`, and `ORIGINAL_DIR` are valid. Writes CSVs under the configured output / leaderboard folders.

---

### 5. Evaluate pretrained Real-ESRGAN (no LoRA)

```bash
python evaluate_realesrgan_only_v2.py
```

Uses `RealESRGANer` + `RealESRGAN_x4plus.pth`; supports periodic JSON checkpointing for long runs. Adjust `RealESRGANOnlyConfig` paths inside the file for your drives.

---

### 6. Evaluate LoRA fine-tuned model

```bash
python evaluate_lora_model.py
python evaluate_lora_model.py --checkpoint ./lora_checkpoint_r32.pth
python evaluate_lora_model.py --checkpoint ./lora_checkpoint_epoch44.pth --max_samples 50
```

Optional: `--base_model`, `--pairs_csv`, `--output_dir`, `--rank`, `--alpha`, `--no_save_images`.

---

## Outputs (typical)

- **Training:** `lora_checkpoint_r32.pth` (or name set in YAML) — contains `model_state`, optimizer, scheduler, epoch, etc.; optional LoRA-only shards if `save_lora_only: true`.
- **Real-ESRGAN eval:** under `OUTPUT_DIR` from config — leaderboards, enhanced images, optional `checkpoint.json`.
- **LoRA eval:** similar structure under `LoRAEvalConfig` output directory (see script defaults).

---

## Project notes

- **Inference** does not classify degradation type; CSV `mode` labels are for **evaluation grouping** only.
- **LoRA** wraps selected `Conv2d` layers: forward is `conv(x) + lora(x)` with frozen base convolutions.
- For a **single-image demo** without full evaluation, see also `enhancement_lora_model.py` (not listed above but related).

---

## Author

FYP project — The Hong Kong Polytechnic University.
