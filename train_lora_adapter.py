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
import platform

# Fix multiprocessing issues on macOS
if platform.system() == 'Darwin':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set

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


# Helper: format time duration
def format_duration(seconds: float) -> str:
    """Format seconds into human-readable string (e.g., '2h 30m 45s')."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return ' '.join(parts)

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

        # resolve path by trying multiple column variants for cross-platform support
        # Priority order: base column -> (P) variant -> (Mac) variant
        # This allows the same CSV to work on Windows (D:\, P:\) and Mac (/Volumes/...)
        def resolve_path(record, base_key):
            # Define all possible column name variants to try
            # Order: primary -> Windows P: variant -> Mac variant
            column_variants = [
                base_key,                    # e.g., "degraded_path" (D:\...)
                f"{base_key}(P)",            # e.g., "degraded_path(P)" (P:\...)
                f"{base_key} (P)",           # e.g., "degraded_path (P)" (P:\...)
                f"{base_key}(Mac)",          # e.g., "degraded_path(Mac)" (/Volumes/...)
                f"{base_key} (Mac)",         # e.g., "degraded_path (Mac)" (/Volumes/...)
                f"{base_key}_mac",           # e.g., "degraded_path_mac"
                f"{base_key}_Mac",           # e.g., "degraded_path_Mac"
            ]
            
            tried_paths = {}
            for col_name in column_variants:
                value = record.get(col_name)
                if isinstance(value, str) and value.strip():
                    path_str = value.strip()
                    p = Path(path_str)
                    tried_paths[col_name] = path_str
                    if p.exists():
                        return str(p)
            
            # nothing found - provide detailed error message
            if not tried_paths:
                raise FileNotFoundError(
                    f"No path columns found for '{base_key}'. "
                    f"Tried columns: {column_variants}. "
                    f"Available columns: {list(record.keys())}"
                )
            raise FileNotFoundError(
                f"No existing file found for '{base_key}'. "
                f"Tried paths:\n" + 
                "\n".join(f"  - {col}: {path}" for col, path in tried_paths.items())
            )

        degraded_path = resolve_path(rec, 'degraded_path')
        target_path = resolve_path(rec, 'target_path')

        # metadata is optional; try to resolve if present
        # Check for any metadata column variant
        metadata_columns = ['metadata_path', 'metadata_path(P)', 'metadata_path (P)', 
                           'metadata_path(Mac)', 'metadata_path (Mac)', 'metadata_path_mac', 'metadata_path_Mac']
        metadata_path = None
        has_metadata_col = any(col in rec for col in metadata_columns)
        if has_metadata_col:
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
    """Load RRDBNet model, with fallback for basicsr compatibility issues."""
    RRDBNet = None
    
    # Method 1: Try direct import from basicsr.archs (avoids problematic data module)
    try:
        import basicsr.archs.rrdbnet_arch as rrdb_module
        RRDBNet = rrdb_module.RRDBNet
    except ImportError:
        pass
    except Exception:
        pass
    
    # Method 2: If basicsr fails, try realesrgan package
    if RRDBNet is None:
        try:
            from realesrgan.archs.rrdbnet_arch import RRDBNet as RRDBNetAlt
            RRDBNet = RRDBNetAlt
        except ImportError:
            pass
    
    # Method 3: Define RRDBNet locally (standalone implementation)
    if RRDBNet is None:
        print("Warning: basicsr/realesrgan not available, using built-in RRDBNet")
        RRDBNet = _get_builtin_rrdbnet()
    
    if RRDBNet is None:
        raise RuntimeError('Could not load RRDBNet. Install basicsr or realesrgan: pip install basicsr')
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # load pretrained weights
    if model_path.exists():
        ckpt = torch.load(str(model_path), map_location='cpu', weights_only=True)
        if 'params_ema' in ckpt:
            state = ckpt['params_ema']
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    return model


def _get_builtin_rrdbnet():
    """Standalone RRDBNet implementation for when basicsr is not available."""
    
    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # upsampling
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            # upsample
            feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out
    
    return RRDBNet


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
    # Support CUDA (NVIDIA), MPS (Apple Silicon), or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple MPS (Metal Performance Shaders)')
    else:
        device = torch.device('cpu')
        print('Using CPU (this will be slow)')
    scale = int(cfg.get('scale', 4))  # Real-ESRGAN default is 4x
    patch_size = int(cfg['patch_size'])
    batch_size = int(cfg['batch_size'])
    
    # MPS memory management: automatically reduce settings if they might cause OOM
    if device.type == 'mps':
        # MPS has limited memory pool (~46GB on M4 Max)
        # RRDB is very memory-hungry, especially with large patches
        max_safe_batch = 4 if patch_size <= 128 else 2 if patch_size <= 192 else 1
        if batch_size > max_safe_batch:
            print(f'⚠️  MPS memory warning: batch_size={batch_size} may cause OOM with patch_size={patch_size}')
            print(f'    Automatically reducing batch_size to {max_safe_batch}')
            batch_size = max_safe_batch
        if patch_size > 192:
            print(f'⚠️  MPS memory warning: patch_size={patch_size} is very large for MPS')
            print(f'    Consider reducing to 128 or 192 if you encounter OOM errors')
    
    print(f'Training config: patch_size={patch_size}, scale={scale}, batch_size={batch_size}')
    print(f'  LR input: {patch_size}x{patch_size}, HR target: {patch_size*scale}x{patch_size*scale}')
    
    ds = PairedDataset(Path(cfg['pairs_csv']), patch_size=patch_size, split='train', 
                       small=(cfg['mode']=='small'), scale=scale)
    # pin_memory is only beneficial for CUDA, not MPS
    # num_workers may cause issues on macOS, reduce if needed
    pin_mem = (device.type == 'cuda')
    num_workers = 4 if device.type == 'cuda' else 2  # Reduce workers on MPS to avoid multiprocessing issues
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_ds = PairedDataset(Path(cfg['pairs_csv']), patch_size=patch_size, split='val', 
                           small=True, scale=scale)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

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
    accumulated_time = 0.0  # Total training time from previous sessions (seconds)
    
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
        # Restore accumulated training time
        accumulated_time = ck.get('total_training_time', 0.0)
        print(f'Resumed from checkpoint epoch {start_epoch}')
        print(f'  Previous training time: {format_duration(accumulated_time)}')

    loss_l1 = nn.L1Loss()

    # prepare AMP GradScaler once if requested and CUDA is available
    # Use new PyTorch 2.0+ API (torch.amp instead of torch.cuda.amp)
    # Note: AMP is only supported on CUDA, not MPS
    scaler = None
    use_amp = cfg.get('amp', False) and device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print('AMP (Mixed Precision) enabled')

    epochs = int(cfg['epochs'])
    session_start_time = time.time()  # Start timer for this training session
    
    # Signal handler for graceful interruption (Ctrl+C)
    import signal
    interrupted = False
    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        session_elapsed = time.time() - session_start_time
        total_time = accumulated_time + session_elapsed
        print(f'\n\n⚠️  Training interrupted!')
        print(f'  This session: {format_duration(session_elapsed)}')
        print(f'  Total training time: {format_duration(total_time)}')
        print('  (Progress saved in last checkpoint)')
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f'\nStarting training from epoch {start_epoch+1} to {epochs}...\n')
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        
        # Progress bar for each epoch
        pbar = tqdm(dl, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
        for i, (lr, hr) in enumerate(pbar):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
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
            
            # Clear cache periodically to prevent memory buildup
            # More frequent clearing on MPS due to memory pool limitations
            clear_interval = 10 if device.type == 'mps' else 50
            if i % clear_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                    # Force synchronization to ensure memory is freed
                    torch.mps.synchronize()
            
            # Intra-epoch checkpoint: save every N iterations (default: 500)
            save_every_n = int(cfg.get('save_every_n_iterations', 500))
            if save_every_n > 0 and (i + 1) % save_every_n == 0:
                session_elapsed = time.time() - session_start_time
                total_time = accumulated_time + session_elapsed
                ck_state = {
                    'epoch': epoch,  # Current epoch (not completed yet)
                    'iteration': i + 1,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                    'total_training_time': total_time,
                }
                ck_path = Path(cfg['checkpoint'])
                save_checkpoint(ck_state, ck_path)
                tqdm.write(f'  💾 Checkpoint saved at iteration {i+1}/{len(dl)} (total time: {format_duration(total_time)})')

        avg_loss = total_loss / len(dl)
        epoch_time = time.time() - epoch_start_time
        session_elapsed = time.time() - session_start_time
        total_time = accumulated_time + session_elapsed
        
        # Step learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.6f} - lr: {current_lr:.2e} - epoch time: {format_duration(epoch_time)} - total: {format_duration(total_time)}')
        else:
            print(f'Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.6f} - epoch time: {format_duration(epoch_time)} - total: {format_duration(total_time)}')

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

        # checkpoint (include total training time)
        session_elapsed = time.time() - session_start_time
        total_time = accumulated_time + session_elapsed
        ck_state = {
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'total_training_time': total_time,  # Accumulated training time in seconds
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

    # Calculate final total time
    session_elapsed = time.time() - session_start_time
    total_time = accumulated_time + session_elapsed
    
    print('\n' + '='*60)
    print('✅ Training finished!')
    print(f'  This session: {format_duration(session_elapsed)}')
    print(f'  Total training time: {format_duration(total_time)}')
    print(f'  Epochs completed: {epochs}')
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_lora.yaml')
    parser.add_argument('--mode', type=str, default='small', choices=['small', 'full'])
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Specific checkpoint file to resume from (overrides config)')
    args = parser.parse_args()

    # load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print('Config not found:', cfg_path)
        sys.exit(1)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # override mode
    cfg['mode'] = args.mode
    if args.mode == 'small':
        cfg['epochs'] = 2
        cfg['batch_size'] = 4
        cfg['patch_size'] = 128
    else:
        cfg['epochs'] = cfg.get('epochs', 50)
    
    # Allow overriding checkpoint path from command line
    if args.checkpoint:
        cfg['checkpoint'] = args.checkpoint
        args.resume = True  # Automatically enable resume if checkpoint specified
        print(f'Using custom checkpoint: {args.checkpoint}')

    # resolve configured absolute paths: prefer given path, otherwise try alternative paths
    # Supports: D:\ (Windows laptop) -> P:\ (other Windows) -> /Volumes/Extreme SSD (Mac)
    import re
    def resolve_cfg_path(path_str):
        try:
            p = Path(path_str)
        except Exception:
            return path_str
        if p.exists():
            return str(p)
        s = str(path_str)
        
        # Extract the relative path part (after drive letter or volume)
        # e.g., "D:/degraded_full_dataset/pairs.csv" -> "degraded_full_dataset/pairs.csv"
        relative_path = None
        m = re.match(r'^([A-Za-z]):[/\\](.*)$', s)
        if m:
            relative_path = m.group(2).replace('\\', '/')
        
        # List of base paths to try (in order)
        base_paths = [
            'D:/',                      # Windows laptop
            'P:/',                      # Other Windows machines
            '/Volumes/Extreme SSD/',    # Mac external SSD
        ]
        
        if relative_path:
            for base in base_paths:
                test_path = Path(base) / relative_path
                if test_path.exists():
                    print(f'  Path resolved: {path_str} -> {test_path}')
                    return str(test_path)
        
        # Legacy fallback: try swapping drive letters directly
        s2 = s.replace('/', '\\')
        if Path(s2).exists():
            return str(Path(s2))
        if m:
            alt = 'P:' + m.group(2)
            if Path(alt).exists():
                return str(Path(alt))
        
        # Return original if nothing works (will fail later with clear error)
        return path_str

    if 'pairs_csv' in cfg:
        cfg['pairs_csv'] = resolve_cfg_path(cfg['pairs_csv'])
    if 'model_path' in cfg:
        cfg['model_path'] = resolve_cfg_path(cfg['model_path'])

    train_loop(cfg, resume=args.resume)
