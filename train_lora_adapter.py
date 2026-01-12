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
import yaml
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

# Example: patching RRDBBlock convs (this is pseudo and must match basicsr's RRDBNet impl)
class RRDBWithLoRA(nn.Module):
    def __init__(self, base_rrdb, rank=8, alpha=1.0, insert_in_blocks=True):
        super().__init__()
        self.base = base_rrdb
        self.rank = rank
        self.alpha = alpha
        self.adapters = nn.ModuleList()
        # traverse modules and add adapters where conv2d exists in residual blocks
        for name, module in self.base.named_modules():
            if isinstance(module, nn.Conv2d):
                # only add adapters to convs inside residual blocks heuristically
                if 'residual' in name or 'rrdb' in name or insert_in_blocks:
                    a = ConvLoRA(module.in_channels, module.out_channels, rank, alpha)
                    self.adapters.append((name, a))
        # store mapping (simple list for this small script)

    def forward(self, x):
        # run base model to get base_out
        base_out = self.base(x)
        # naive: sum adapters applied to input and add to base_out (illustrative)
        delta = 0
        for nm, adapter in self.adapters:
            # naive application (not aligned with base conv positions) - for demo only
            delta = delta + adapter(x)
        return base_out + delta


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
        degraded = cv2.imread(rec['degraded_path'])
        gt = cv2.imread(rec['target_path'])
        if degraded is None or gt is None:
            raise RuntimeError('Failed to read images')
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

    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        total_loss = 0.0
        for i, (lr, hr) in enumerate(dl):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg['amp'] and device.type=='cuda'):
                out = model(lr)
                loss = loss_l1(out, hr)
            if cfg['amp'] and device.type=='cuda':
                scaler = torch.cuda.amp.GradScaler()
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

    train_loop(cfg, resume=args.resume)
