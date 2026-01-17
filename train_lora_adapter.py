#!/usr/bin/env python3
"""
LoRA adapter training script for RRDBNet (Real-ESRGAN)
- Single-adapter joint training (motion blur + low-light)
- Small-scale test mode and full training mode
- Checkpointing and resume support

Usage examples:
  # small test (debug)
  python train_lora_adapter.py --config configs/train_lora.yaml --mode small

  # full training
  python train_lora_adapter.py --config configs/train_lora.yaml --mode full

"""
import os
import sys
import argparse
try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML (import yaml). Install with: pip install pyyaml")
    raise
import time
from pathlib import Path
from typing import Dict, Any, List

import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
from tqdm import tqdm

# Optional: LPIPS for perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# Helper: compute PSNR
def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute PSNR between two image tensors (B, C, H, W) in [0, max_val]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

# Minimal LoRA adapter for Conv2d
class ConvLoRA(nn.Module):
    def __init__(self, in_channels, out_channels, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # down and up are 1x1 convs
        self.down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        # zero-init up for stability
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x)) * self.alpha


class Conv2dWithLoRA(nn.Module):
    """Wrap a nn.Conv2d with a small LoRA adapter (down/up 1x1 convs).
    The original conv's parameters are frozen; only the adapter params are trainable.
    """
    def __init__(self, conv: nn.Conv2d, rank=8, alpha=1.0):
        super().__init__()
        # keep a reference to the original conv module
        self.conv = conv
        # freeze base conv params
        for p in self.conv.parameters():
            p.requires_grad = False
        # adapter applied to activations (1x1 down/up)
        self.lora = ConvLoRA(conv.in_channels, conv.out_channels, rank=rank, alpha=alpha)

    def forward(self, x):
        # base conv output + LoRA adapter output
        return self.conv(x) + self.lora(x)

# Example: patching RRDBBlock convs (this is pseudo and must match basicsr's RRDBNet impl)
class RRDBWithLoRA(nn.Module):
    def __init__(self, base_rrdb, rank=8, alpha=1.0, insert_in_blocks=True):
        super().__init__()
        self.base = base_rrdb
        self.rank = rank
        self.alpha = alpha
        # Replace target Conv2d modules in-place with Conv2dWithLoRA wrapper so the
        # LoRA adapter is applied at the correct conv locations. Only adapter
        # parameters (inside the wrappers) will be trainable.
        # We iterate over a copy of named_modules to avoid mutation issues.
        for name, module in list(self.base.named_modules()):
            if isinstance(module, nn.Conv2d):
                if 'residual' in name or 'rrdb' in name or insert_in_blocks:
                    # locate parent module and replace child in parent._modules
                    parts = name.split('.')
                    parent = self.base
                    for p in parts[:-1]:
                        parent = parent._modules.get(p)
                        if parent is None:
                            break
                    else:
                        child_name = parts[-1]
                        # ensure current child matches expected module
                        if child_name in parent._modules and parent._modules[child_name] is module:
                            parent._modules[child_name] = Conv2dWithLoRA(module, rank=rank, alpha=alpha)

    def forward(self, x):
        # Delegates to the patched base model; Conv2dWithLoRA wrappers will
        # produce base_conv(x) + adapter(x) for replaced convs.
        return self.base(x)


class PairedDataset(Dataset):
    """Dataset for paired degraded/clean images.
    
    For Real-ESRGAN (4x upscaler):
    - degraded image is downsampled to patch_size (input to model)
    - target image is cropped to patch_size * scale (output target)
    - Model outputs patch_size * 4, which matches the target
    """
    def __init__(self, pairs_csv: Path, patch_size: int = 256, split: str = 'train', small: bool = False, scale: int = 4):
        self.df = None
        self.pairs = []
        self.patch_size = patch_size  # LR patch size (input size)
        self.scale = scale  # upscaling factor (4 for Real-ESRGAN)
        self.hr_size = patch_size * scale  # HR patch size (target size)
        self.small = small

        if not pairs_csv.exists():
            raise FileNotFoundError(f"pairs csv not found: {pairs_csv}")
        import pandas as pd
        self.df = pd.read_csv(pairs_csv)
        if 'split' in self.df.columns:
            if split:
                self.df = self.df[self.df['split'] == split]
        self.pairs = self.df.to_dict('records')
        if self.small:
            self.pairs = self.pairs[:64]

        # transforms
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def _random_crop_pair(self, img_degraded, img_gt):
        """Crop paired patches for super-resolution training.
        
        Returns:
            lr_crop: patch_size x patch_size (input to model)
            hr_crop: hr_size x hr_size (target, = patch_size * scale)
        """
        h, w = img_gt.shape[:2]
        # Ensure image is large enough for HR crop
        if h < self.hr_size or w < self.hr_size:
            # Scale up to minimum size
            scale_factor = max(self.hr_size / h, self.hr_size / w)
            new_h, new_w = int(h * scale_factor) + 1, int(w * scale_factor) + 1
            img_gt = cv2.resize(img_gt, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            img_degraded = cv2.resize(img_degraded, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = img_gt.shape[:2]
        
        # Random crop position for HR
        x = random.randint(0, w - self.hr_size)
        y = random.randint(0, h - self.hr_size)
        
        # Crop HR target
        hr_crop = img_gt[y:y+self.hr_size, x:x+self.hr_size]
        
        # Crop degraded at same position, then downsample to LR size
        degraded_crop = img_degraded[y:y+self.hr_size, x:x+self.hr_size]
        lr_crop = cv2.resize(degraded_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        
        return lr_crop, hr_crop

    def __getitem__(self, idx):
        rec = self.pairs[idx]

        # resolve primary key first, then fallback to "(P)" variants
        def resolve_path(record, base_key):
            primary = record.get(base_key)
            if isinstance(primary, str):
                p = Path(primary.strip())
                if p.exists():
                    return str(p)
            # try explicit "(P)" column names (two common variants)
            alt_keys = [f"{base_key}(P)", f"{base_key} (P)"]
            for k in alt_keys:
                v = record.get(k)
                if isinstance(v, str):
                    p2 = Path(v.strip())
                    if p2.exists():
                        return str(p2)
            # nothing found
            tried = {base_key: record.get(base_key), alt_keys[0]: record.get(alt_keys[0]), alt_keys[1]: record.get(alt_keys[1])}
            raise FileNotFoundError(f"No existing file found for keys {list(tried.keys())}. Values: {tried}")

        degraded_path = resolve_path(rec, 'degraded_path')
        target_path = resolve_path(rec, 'target_path')

        # metadata is optional; try to resolve if present
        metadata_path = None
        if ('metadata_path' in rec) or ('metadata_path(P)' in rec) or ('metadata_path (P)' in rec):
            try:
                metadata_path = resolve_path(rec, 'metadata_path')
            except FileNotFoundError:
                metadata_path = None

        degraded = cv2.imread(degraded_path)
        gt = cv2.imread(target_path)
        if degraded is None or gt is None:
            raise RuntimeError(f'Failed to read images. degraded: {degraded_path}, target: {target_path}')

        # Ensure degraded and gt have the same size
        if degraded.shape != gt.shape:
            degraded = cv2.resize(degraded, (gt.shape[1], gt.shape[0]))
        
        # Always use random crop for training to get proper LR/HR pair
        # LR (input): patch_size x patch_size
        # HR (target): patch_size*scale x patch_size*scale
        degraded, gt = self._random_crop_pair(degraded, gt)
        
        # BGR->RGB
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        degraded_t = self.to_tensor(degraded)
        gt_t = self.to_tensor(gt)
        return degraded_t, gt_t


def freeze_base_params(model):
    """Freeze base model parameters but keep LoRA adapter params trainable."""
    for name, p in model.base.named_parameters():
        # Don't freeze LoRA adapter parameters
        if 'lora' in name or ('down' in name and 'weight' in name) or ('up' in name and 'weight' in name):
            p.requires_grad = True
        else:
            p.requires_grad = False


def load_base_rrdb(model_path: Path):
    # create RRDB from basicsr
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except Exception as e:
        raise RuntimeError('basicsr is required')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # load pretrained weights
    if model_path.exists():
        import torch
        ckpt = torch.load(str(model_path), map_location='cpu', weights_only=True)
        if 'params_ema' in ckpt:
            state = ckpt['params_ema']
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    return model


def save_checkpoint(state: Dict[str, Any], path: Path):
    torch.save(state, str(path))


def load_checkpoint(path: Path):
    if not path.exists():
        return None
    return torch.load(str(path), map_location='cpu', weights_only=False)


def build_model_and_optimizer(cfg):
    base = load_base_rrdb(Path(cfg['model_path']))
    wrapped = RRDBWithLoRA(base, rank=int(cfg['rank']), alpha=float(cfg['alpha']))
    # freeze base
    freeze_base_params(wrapped)
    
    # collect adapter params - look for LoRA adapter parameters
    adapter_params = []
    for n, p in wrapped.named_parameters():
        # LoRA adapter params are in 'lora.down' and 'lora.up' modules
        if 'lora' in n and ('down' in n or 'up' in n):
            p.requires_grad = True  # ensure requires_grad is set
            adapter_params.append(p)
    
    if len(adapter_params) == 0:
        raise RuntimeError(f"No LoRA adapter parameters found! Check RRDBWithLoRA patching.")
    
    print(f"Found {len(adapter_params)} LoRA adapter parameters to train")
    total_params = sum(p.numel() for p in adapter_params)
    print(f"Total trainable params: {total_params:,}")
    
    lr = float(cfg['lr'])
    optimizer = torch.optim.Adam(adapter_params, lr=lr, betas=(0.9, 0.99))
    
    # Learning rate scheduler (like Real-ESRGAN reference)
    scheduler = None
    if cfg.get('use_scheduler', False):
        milestones = cfg.get('lr_milestones', [cfg.get('epochs', 50) // 2])
        gamma = cfg.get('lr_gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    return wrapped, optimizer, scheduler


def train_loop(cfg, resume: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale = int(cfg.get('scale', 4))  # Real-ESRGAN default is 4x
    patch_size = int(cfg['patch_size'])
    batch_size = int(cfg['batch_size'])
    
    print(f'Training config: patch_size={patch_size}, scale={scale}, batch_size={batch_size}')
    print(f'  LR input: {patch_size}x{patch_size}, HR target: {patch_size*scale}x{patch_size*scale}')
    
    ds = PairedDataset(Path(cfg['pairs_csv']), patch_size=patch_size, split='train', 
                       small=(cfg['mode']=='small'), scale=scale)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ds = PairedDataset(Path(cfg['pairs_csv']), patch_size=patch_size, split='val', 
                           small=True, scale=scale)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model, optimizer, scheduler = build_model_and_optimizer(cfg)
    model.to(device)

    # Optional: Perceptual loss (LPIPS) for better visual quality
    lpips_loss_fn = None
    use_perceptual = cfg.get('use_perceptual_loss', False)
    perceptual_weight = cfg.get('perceptual_weight', 0.1)
    if use_perceptual and LPIPS_AVAILABLE:
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
        lpips_loss_fn.eval()
        for p in lpips_loss_fn.parameters():
            p.requires_grad = False
        print(f'Perceptual loss (LPIPS) enabled, weight={perceptual_weight}')
    elif use_perceptual and not LPIPS_AVAILABLE:
        print('Warning: use_perceptual_loss=True but lpips not installed. Using L1 only.')

    start_epoch = 0
    if resume and Path(cfg['checkpoint']).exists():
        ck = load_checkpoint(Path(cfg['checkpoint']))
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        # Ensure optimizer state tensors are on the correct device (GPU/CPU)
        # When checkpoints are saved/loaded with map_location='cpu', optimizer
        # internal tensors remain on CPU. Move them to `device` to avoid
        # device-mismatch runtime errors when resuming on GPU.
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # Restore scheduler state if available
        if scheduler is not None and ck.get('scheduler_state') is not None:
            scheduler.load_state_dict(ck['scheduler_state'])
        start_epoch = ck.get('epoch', 0)
        print('Resumed from checkpoint epoch', start_epoch)

    loss_l1 = nn.L1Loss()

    # prepare AMP GradScaler once if requested and CUDA is available
    # Use new PyTorch 2.0+ API (torch.amp instead of torch.cuda.amp)
    scaler = None
    if cfg.get('amp', False) and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')

    epochs = int(cfg['epochs'])
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        
        # Progress bar for each epoch
        pbar = tqdm(dl, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
        for i, (lr, hr) in enumerate(pbar):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                out = model(lr)
                # L1 loss
                l1_val = loss_l1(out, hr)
                loss = l1_val
                # Add perceptual loss if enabled
                if lpips_loss_fn is not None:
                    # LPIPS expects input in [-1, 1], so scale from [0, 1]
                    out_scaled = out * 2 - 1
                    hr_scaled = hr * 2 - 1
                    perceptual_val = lpips_loss_fn(out_scaled, hr_scaled).mean()
                    loss = loss + perceptual_weight * perceptual_val
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dl)
        
        # Step learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.6f} - lr: {current_lr:.2e}')
        else:
            print(f'Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.6f}')

        # validation with PSNR
        model.eval()
        val_psnr_list = []
        with torch.no_grad():
            for j, (lr, hr) in enumerate(val_dl):
                lr = lr.to(device)
                hr = hr.to(device)
                out = model(lr)
                # Clamp output to valid range
                out = out.clamp(0, 1)
                # Compute PSNR
                psnr = compute_psnr(out, hr, max_val=1.0)
                val_psnr_list.append(psnr)
                if j >= 9:  # limit validation samples
                    break
        if val_psnr_list:
            avg_psnr = np.mean(val_psnr_list)
            print(f'  Validation PSNR: {avg_psnr:.2f} dB')

        # checkpoint
        ck_state = {
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        }
        ck_path = Path(cfg['checkpoint'])
        # always save latest
        save_checkpoint(ck_state, ck_path)
        # optionally save an epoched checkpoint (safer, allows rollbacks)
        if cfg.get('epoched_checkpoints', False):
            ep_path = ck_path.parent / f"{ck_path.stem}_epoch{epoch+1}{ck_path.suffix}"
            save_checkpoint(ck_state, ep_path)
            # retention: keep only the most recent N epoched checkpoints
            keep_n = int(cfg.get('keep_last', 0))
            if keep_n > 0:
                pattern = f"{ck_path.stem}_epoch*{ck_path.suffix}"
                files = sorted(ck_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime)
                if len(files) > keep_n:
                    for old in files[:-keep_n]:
                        try:
                            old.unlink()
                        except Exception:
                            pass
        # optionally save LoRA-only weights (much smaller)
        if cfg.get('save_lora_only', False):
            model_state = ck_state['model_state']
            lora_state = {k: v for k, v in model_state.items() if ('down' in k or 'up' in k or 'lora' in k)}
            lora_path = ck_path.parent / f"{ck_path.stem}_lora_epoch{epoch+1}{ck_path.suffix}"
            try:
                torch.save({'epoch': epoch+1, 'lora_state': lora_state}, str(lora_path))
            except Exception:
                pass
        print('Checkpoint saved', cfg['checkpoint'])

    print('Training finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_lora.yaml')
    parser.add_argument('--mode', type=str, default='small', choices=['small', 'full'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print('Config not found:', cfg_path)
        sys.exit(1)
    cfg = yaml.safe_load(open(cfg_path))
    # override mode
    cfg['mode'] = args.mode
    if args.mode == 'small':
        cfg['epochs'] = 2
        cfg['batch_size'] = 4
        cfg['patch_size'] = 128
    else:
        cfg['epochs'] = cfg.get('epochs', 50)

    # resolve configured absolute paths: prefer given path, otherwise try swapping drive to P:
    import re
    def resolve_cfg_path(path_str):
        try:
            p = Path(path_str)
        except Exception:
            return path_str
        if p.exists():
            return str(p)
        s = str(path_str)
        # normalize slashes
        s2 = s.replace('/', '\\')
        if Path(s2).exists():
            return str(Path(s2))
        # if path has a drive letter, replace it with P:
        m = re.match(r'^([A-Za-z]):(.*)$', s2)
        if m:
            alt = 'P:' + m.group(2)
            if Path(alt).exists():
                return str(Path(alt))
        # last attempt: swap leading drive to P: even if original had no drive
        if len(s2) > 2 and s2[1] == ':':
            alt2 = 'P:' + s2[2:]
            if Path(alt2).exists():
                return str(Path(alt2))
        return path_str

    if 'pairs_csv' in cfg:
        cfg['pairs_csv'] = resolve_cfg_path(cfg['pairs_csv'])
    if 'model_path' in cfg:
        cfg['model_path'] = resolve_cfg_path(cfg['model_path'])

    train_loop(cfg, resume=args.resume)
