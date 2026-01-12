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
    def __init__(self, pairs_csv: Path, patch_size: int = 256, split: str = 'train', small: bool = False):
        self.df = None
        self.pairs = []
        self.patch_size = patch_size
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

    def _random_crop_pair(self, img_lr, img_hr):
        h, w = img_lr.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            img_lr = cv2.resize(img_lr, (max(self.patch_size, w), max(self.patch_size, h)))
            img_hr = cv2.resize(img_hr, (img_lr.shape[1], img_lr.shape[0]))
            h, w = img_lr.shape[:2]
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        lr_crop = img_lr[y:y+self.patch_size, x:x+self.patch_size]
        hr_crop = img_hr[y:y+self.patch_size, x:x+self.patch_size]
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

        if degraded.shape != gt.shape:
            degraded = cv2.resize(degraded, (gt.shape[1], gt.shape[0]))
        if self.small:
            degraded, gt = self._random_crop_pair(degraded, gt)
        # BGR->RGB
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        degraded_t = self.to_tensor(degraded)
        gt_t = self.to_tensor(gt)
        return degraded_t, gt_t


def freeze_base_params(model):
    for p in model.base.parameters():
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
        ckpt = torch.load(str(model_path), map_location='cpu')
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
    return torch.load(str(path), map_location='cpu')


def build_model_and_optimizer(cfg):
    base = load_base_rrdb(Path(cfg['model_path']))
    wrapped = RRDBWithLoRA(base, rank=cfg['rank'], alpha=cfg['alpha'])
    # freeze base
    freeze_base_params(wrapped)
    # collect adapter params
    adapter_params = [p for n, p in wrapped.named_parameters() if 'down' in n or 'up' in n]
    optimizer = torch.optim.Adam(adapter_params, lr=cfg['lr'], betas=(0.9, 0.99))
    return wrapped, optimizer


def train_loop(cfg, resume: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = PairedDataset(Path(cfg['pairs_csv']), patch_size=cfg['patch_size'], split='train', small=(cfg['mode']=='small'))
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_ds = PairedDataset(Path(cfg['pairs_csv']), patch_size=cfg['patch_size'], split='val', small=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model, optimizer = build_model_and_optimizer(cfg)
    model.to(device)

    start_epoch = 0
    if resume and Path(cfg['checkpoint']).exists():
        ck = load_checkpoint(Path(cfg['checkpoint']))
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        start_epoch = ck.get('epoch', 0)
        print('Resumed from checkpoint epoch', start_epoch)

    loss_l1 = nn.L1Loss()

    # prepare AMP GradScaler once if requested and CUDA is available
    scaler = None
    if cfg.get('amp', False) and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        total_loss = 0.0
        for i, (lr, hr) in enumerate(dl):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = model(lr)
                loss = loss_l1(out, hr)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        print(f'Epoch {epoch+1}/{cfg["epochs"]} - loss: {avg_loss:.6f}')

        # validation (simple)
        model.eval()
        with torch.no_grad():
            for j, (lr, hr) in enumerate(val_dl):
                lr = lr.to(device)
                hr = hr.to(device)
                out = model(lr)
                # compute PSNR or LPIPS later
                break

        # checkpoint
        ck_state = {
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }
        save_checkpoint(ck_state, Path(cfg['checkpoint']))
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
