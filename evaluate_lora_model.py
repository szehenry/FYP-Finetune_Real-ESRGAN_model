#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA-finetuned Real-ESRGAN 評估腳本
====================================

與 evaluate_realesrgan_only_v2.py 相同的評估流程和輸出格式：
1. 只評估 test split 圖像
2. 計算相同的指標 (PSNR, SSIM, LPIPS, 無參考指標)
3. 生成相同格式的排行榜 (per degradation type)
4. 支援 checkpoint 恢復

用法:
  # 評估最新 checkpoint (lora_checkpoint.pth)
  python evaluate_lora_model.py
  
  # 評估特定 epoch 的 checkpoint
  python evaluate_lora_model.py --checkpoint ./lora_checkpoint_epoch44.pth
  
  # 快速測試 (限制樣本數)
  python evaluate_lora_model.py --max_samples 50

作者：FYP Project
日期：2025-01
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import random
import gc
import re
import argparse

warnings.filterwarnings('ignore')

# 設置隨機種子
random.seed(42)
np.random.seed(42)

# PyTorch
import torch
import torch.nn as nn

# 檢查設備
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# 導入指標計算器 (從 baseline_evaluation 模組)
try:
    from baseline_evaluation import MetricsCalculator
    print("✓ MetricsCalculator 導入成功")
    METRICS_CALCULATOR_AVAILABLE = True
except ImportError:
    print("⚠️  無法導入 MetricsCalculator，將使用內建指標計算")
    METRICS_CALCULATOR_AVAILABLE = False

# LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
    print("✓ LPIPS 導入成功")
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS 不可用")

# SSIM
try:
    from skimage.metrics import structural_similarity as compute_ssim_skimage
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


# ==================== 路徑解析 ====================

# 跨平台基礎路徑（優先順序）
BASE_PATHS = [
    'D:/',                      # Windows 本機
    'P:/',                      # 其他 Windows
    '/Volumes/Extreme SSD/',    # Mac 外接 SSD
]

def resolve_path(path_str: str) -> str:
    """
    跨平台路徑解析 (與 train_lora_adapter.py 相同邏輯)
    嘗試順序: 原始路徑 -> D:/ -> P:/ -> /Volumes/Extreme SSD/
    """
    if not path_str:
        return path_str
    
    p = Path(path_str)
    if p.exists():
        return str(p)
    
    s = str(path_str)
    
    # 提取相對路徑 (去除驅動器號)
    m = re.match(r'^([A-Za-z]):[/\\](.*)$', s)
    if m:
        relative_path = m.group(2).replace('\\', '/')
        
        for base in BASE_PATHS:
            test_path = Path(base) / relative_path
            if test_path.exists():
                return str(test_path)
    
    # 返回原始路徑 (讓後續報錯)
    return path_str


def get_output_base_path() -> Path:
    """
    獲取輸出基礎路徑
    優先順序: D:/ -> P:/ -> /Volumes/Extreme SSD/
    """
    for base in BASE_PATHS:
        base_path = Path(base)
        if base_path.exists():
            print(f"  📂 輸出基礎路徑: {base}")
            return base_path
    
    # 如果都不存在，使用當前目錄
    print("  ⚠️  未找到外部儲存，使用當前目錄")
    return Path(".")


# ==================== 模型定義 (與 train_lora_adapter.py 相同) ====================

class ConvLoRA(nn.Module):
    def __init__(self, in_channels, out_channels, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x)) * self.alpha


class Conv2dWithLoRA(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank=8, alpha=1.0):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.lora = ConvLoRA(conv.in_channels, conv.out_channels, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.conv(x) + self.lora(x)


class RRDBWithLoRA(nn.Module):
    def __init__(self, base_rrdb, rank=8, alpha=1.0, insert_in_blocks=True):
        super().__init__()
        self.base = base_rrdb
        self.rank = rank
        self.alpha = alpha
        for name, module in list(self.base.named_modules()):
            if isinstance(module, nn.Conv2d):
                if 'residual' in name or 'rrdb' in name or insert_in_blocks:
                    parts = name.split('.')
                    parent = self.base
                    for p in parts[:-1]:
                        parent = parent._modules.get(p)
                        if parent is None:
                            break
                    else:
                        child_name = parts[-1]
                        if child_name in parent._modules and parent._modules[child_name] is module:
                            parent._modules[child_name] = Conv2dWithLoRA(module, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.base(x)


# Built-in RRDBNet
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
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ==================== 指標計算 (內建版本) ====================

class BuiltinMetricsCalculator:
    """內建指標計算器 (當 baseline_evaluation 不可用時)"""
    
    def __init__(self):
        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
                self.lpips_fn.eval()
            except:
                pass
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算 PSNR"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((255.0 ** 2) / mse)
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算 SSIM"""
        if not SSIM_AVAILABLE:
            return 0.0
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray, img2_gray = img1, img2
        return compute_ssim_skimage(img1_gray, img2_gray)
    
    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """計算 LPIPS"""
        if self.lpips_fn is None:
            return None
        try:
            # 轉換為 tensor
            t1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
            t2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
            t1, t2 = t1.to(DEVICE), t2.to(DEVICE)
            
            with torch.no_grad():
                lpips_val = self.lpips_fn(t1, t2).item()
            return lpips_val
        except:
            return None
    
    def compute_no_reference_metrics(self, img: np.ndarray) -> Dict:
        """計算無參考指標"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Laplacian variance (銳度)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        var_laplacian = laplacian.var()
        
        # Tenengrad (銳度)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(gx**2 + gy**2)
        
        # Blur extent (模糊程度)
        blur_extent = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_extent = 1.0 / (blur_extent + 1e-6)  # 反轉，越高越模糊
        
        # Luminance (亮度)
        luminance = np.mean(gray)
        
        # Contrast (對比度)
        contrast = np.std(gray)
        
        # Entropy (熵)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        # Gradient mean
        gradient_mean = np.mean(np.sqrt(gx**2 + gy**2))
        
        return {
            'var_laplacian': float(var_laplacian),
            'tenengrad': float(tenengrad),
            'blur_extent': float(blur_extent),
            'luminance': float(luminance),
            'contrast': float(contrast),
            'entropy': float(entropy),
            'gradient_mean': float(gradient_mean),
        }
    
    def evaluate_with_reference(self, enhanced: np.ndarray, original: np.ndarray, 
                                calculate_lpips: bool = True) -> Dict:
        """計算所有指標"""
        metrics = self.compute_no_reference_metrics(enhanced)
        metrics['psnr'] = self.compute_psnr(enhanced, original)
        metrics['ssim'] = self.compute_ssim(enhanced, original)
        
        if calculate_lpips:
            lpips_val = self.compute_lpips(enhanced, original)
            metrics['lpips'] = lpips_val if lpips_val is not None else float('nan')
        else:
            metrics['lpips'] = float('nan')
        
        return metrics


# ==================== 配置 ====================

class LoRAEvalConfig:
    """LoRA 評估配置"""
    
    # 路徑配置 (使用 resolve_path 解析)
    PAIRS_CSV = "D:/degraded_full_dataset/pairs.csv"
    BASE_MODEL_PATH = "D:/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"
    
    # LoRA 配置
    LORA_RANK = 16
    LORA_ALPHA = 1.0
    
    # 輸出配置 (動態解析: D:/ -> P:/ -> /Volumes/Extreme SSD/)
    # 會在 main() 中設置為: get_output_base_path() / "lora_eval_results"
    OUTPUT_DIR = None  # 延遲初始化
    LEADERBOARD_DIR = None
    ENHANCED_DIR = None
    
    @classmethod
    def init_output_dirs(cls, custom_output_dir: str = None):
        """初始化輸出目錄（跨平台）"""
        if custom_output_dir:
            cls.OUTPUT_DIR = Path(custom_output_dir)
        else:
            base = get_output_base_path()
            cls.OUTPUT_DIR = base / "lora_r32_eval_results"
        
        cls.LEADERBOARD_DIR = cls.OUTPUT_DIR / "leaderboards"
        cls.ENHANCED_DIR = cls.OUTPUT_DIR / "enhanced_images"
    
    # 退化類型映射 (與 v2 相同)
    DEGRADATION_TYPE_MAP = {
        'global_blur': 'Drone Motion Blur',
        'object_blur': 'Object Motion Blur',
        'low_light': 'Low-light'
    }
    
    # LPIPS 抽樣配置 (與 v2 相同)
    LPIPS_SAMPLE_SIZE = 700
    LPIPS_ENABLED = True
    
    # Checkpoint 配置
    CHECKPOINT_ENABLED = True
    CHECKPOINT_INTERVAL = 100
    
    # 保存增強圖像
    SAVE_ENHANCED_IMAGES = True
    ENHANCED_IMAGE_FORMAT = "png"
    
    # Tile 處理配置（自動根據設備調整）
    # MPS: 自動使用 128
    # CUDA: 自動使用 256（更快）
    TILE_SIZE = None  # None = 自動選擇（CUDA:256, MPS:128）
    TILE_PAD = 32  # padding 減少接縫
    TILE_BLEND = True  # 啟用 tile 混合
    
    @classmethod
    def get_tile_size(cls, device: str) -> int:
        """根據設備返回最佳 tile size"""
        if device == 'cuda':
            return 256  # CUDA 可用更大的 tile
        else:
            return 128  # MPS 用保守設定


# ==================== LoRA 模型載入 ====================

def load_lora_model(checkpoint_path: Path, base_model_path: Path, device: str,
                    rank: int = 16, alpha: float = 1.0):
    """載入 LoRA 微調模型"""
    print(f"\n🔧 載入 LoRA 模型:")
    print(f"  Base model: {base_model_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Rank: {rank}, Alpha: {alpha}")
    
    # 創建基礎模型
    base_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # 載入基礎權重
    if base_model_path.exists():
        base_ckpt = torch.load(str(base_model_path), map_location='cpu', weights_only=True)
        if 'params_ema' in base_ckpt:
            base_model.load_state_dict(base_ckpt['params_ema'], strict=False)
        else:
            base_model.load_state_dict(base_ckpt, strict=False)
        print("  ✓ 基礎權重載入成功")
    else:
        print(f"  ⚠️  基礎模型不存在: {base_model_path}")
    
    # 包裝 LoRA
    model = RRDBWithLoRA(base_model, rank=rank, alpha=alpha)
    
    # 載入 LoRA checkpoint
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'], strict=False)
        epoch = ckpt.get('epoch', '?')
        training_time = ckpt.get('total_training_time', 0) / 3600
        print(f"  ✓ LoRA checkpoint 載入成功 (Epoch {epoch}, 訓練時間: {training_time:.1f}h)")
    else:
        print(f"  ❌ Checkpoint 不存在: {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    return model


def enhance_image(model, img: np.ndarray, device: str, tile_size: int = 256, tile_pad: int = 32) -> np.ndarray:
    """使用 LoRA 模型增強圖像（帶平滑混合的 Tile 處理）
    
    Args:
        model: LoRA 模型
        img: BGR 圖像 (numpy array)
        device: 'cuda' 或 'mps'
        tile_size: Tile 大小（MPS 用 128，CUDA 用 256-512）
        tile_pad: Tile 邊緣填充（越大混合越平滑）
    
    Returns:
        增強後的 BGR 圖像
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    original_h, original_w = h, w
    
    # MPS 兼容性
    if device == 'mps':
        tile_size = 128
        new_h = ((h + 63) // 64) * 64
        new_w = ((w + 63) // 64) * 64
        if new_h != h or new_w != w:
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w
    
    # 決定是否使用 tile 處理
    if device == 'mps':
        use_tile = True
    else:
        use_tile = (h > 512 or w > 512)
    
    scale = 4
    
    def process_tensor(img_tensor):
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            output = output.clamp(0, 1)
        return output
    
    if not use_tile:
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        output = process_tensor(img_tensor)
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        return output_bgr
    
    # ========== 帶混合的 Tile 處理 ==========
    output_h, output_w = h * scale, w * scale
    
    # 輸出累積器和權重累積器（用於混合）
    output_acc = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight_acc = np.zeros((output_h, output_w, 1), dtype=np.float32)
    
    # 使用重疊的 tile（overlap = tile_pad）
    step = tile_size  # 步長 = tile_size（重疊區域由 pad 處理）
    
    # 計算 tile 位置
    y_positions = list(range(0, h - tile_size + 1, step))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    
    x_positions = list(range(0, w - tile_size + 1, step))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    
    # 創建混合權重（中心權重高，邊緣權重低）
    def create_blend_weight(size, pad):
        """創建漸變混合權重"""
        weight = np.ones((size, size), dtype=np.float32)
        # 邊緣漸變
        for i in range(pad):
            factor = (i + 1) / (pad + 1)
            weight[i, :] *= factor
            weight[size - 1 - i, :] *= factor
            weight[:, i] *= factor
            weight[:, size - 1 - i] *= factor
        return weight
    
    blend_weight_lr = create_blend_weight(tile_size + 2 * tile_pad, tile_pad)
    
    for y_start in y_positions:
        for x_start in x_positions:
            # 擴展範圍（加 padding）
            y_start_pad = max(0, y_start - tile_pad)
            y_end_pad = min(h, y_start + tile_size + tile_pad)
            x_start_pad = max(0, x_start - tile_pad)
            x_end_pad = min(w, x_start + tile_size + tile_pad)
            
            # 提取 tile（帶 padding）
            tile = img_rgb[y_start_pad:y_end_pad, x_start_pad:x_end_pad].copy()
            th, tw = tile.shape[:2]
            
            # MPS 尺寸對齊
            orig_th, orig_tw = th, tw
            if device == 'mps':
                new_th = ((th + 63) // 64) * 64
                new_tw = ((tw + 63) // 64) * 64
                if new_th != th or new_tw != tw:
                    tile = cv2.resize(tile, (new_tw, new_th), interpolation=cv2.INTER_CUBIC)
            
            # 處理 tile
            tile_tensor = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tile_output = process_tensor(tile_tensor)
            tile_output_np = tile_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 縮放回原尺寸
            expected_h = orig_th * scale
            expected_w = orig_tw * scale
            if tile_output_np.shape[0] != expected_h or tile_output_np.shape[1] != expected_w:
                tile_output_np = cv2.resize(tile_output_np, (expected_w, expected_h), interpolation=cv2.INTER_CUBIC)
            
            # 創建此 tile 的混合權重
            tile_weight = np.ones((expected_h, expected_w, 1), dtype=np.float32)
            
            # 邊緣漸變（軟邊界）
            fade_size = tile_pad * scale
            for i in range(min(fade_size, expected_h // 2)):
                factor = (i + 1) / (fade_size + 1)
                tile_weight[i, :, 0] *= factor
                tile_weight[expected_h - 1 - i, :, 0] *= factor
            for i in range(min(fade_size, expected_w // 2)):
                factor = (i + 1) / (fade_size + 1)
                tile_weight[:, i, 0] *= factor
                tile_weight[:, expected_w - 1 - i, 0] *= factor
            
            # 累積到輸出
            out_y_start = y_start_pad * scale
            out_y_end = y_end_pad * scale
            out_x_start = x_start_pad * scale
            out_x_end = x_end_pad * scale
            
            output_acc[out_y_start:out_y_end, out_x_start:out_x_end] += tile_output_np * tile_weight
            weight_acc[out_y_start:out_y_end, out_x_start:out_x_end] += tile_weight
            
            # 清理
            del tile_tensor, tile_output
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
    
    # 歸一化（加權平均）
    weight_acc = np.maximum(weight_acc, 1e-8)  # 避免除以零
    output_np = output_acc / weight_acc
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return output_bgr


# ==================== 主評估器 ====================

class LoRAEvaluator:
    """LoRA 模型評估器"""
    
    def __init__(self, checkpoint_path: str, config: LoRAEvalConfig):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.device = DEVICE
        
        # 解析路徑
        self.base_model_path = Path(resolve_path(config.BASE_MODEL_PATH))
        self.pairs_csv_path = Path(resolve_path(config.PAIRS_CSV))
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        
        # 載入模型
        self.model = load_lora_model(
            self.checkpoint_path, 
            self.base_model_path,
            self.device,
            rank=config.LORA_RANK,
            alpha=config.LORA_ALPHA
        )
        
        if self.model is None:
            print("❌ 模型載入失敗")
            sys.exit(1)
        
        # 初始化指標計算器
        if METRICS_CALCULATOR_AVAILABLE:
            self.calculator = MetricsCalculator()
        else:
            self.calculator = BuiltinMetricsCalculator()
        
        # 載入配對信息
        self.pairs_df = self.load_pairs()
    
    def load_pairs(self) -> pd.DataFrame:
        """載入 pairs.csv"""
        if not self.pairs_csv_path.exists():
            print(f"❌ 配對文件不存在: {self.pairs_csv_path}")
            return pd.DataFrame()
        
        print(f"\n📁 載入配對信息: {self.pairs_csv_path}")
        df = pd.read_csv(self.pairs_csv_path)
        print(f"✓ 載入 {len(df)} 對配對")
        
        # 只使用 test split
        if 'split' in df.columns:
            df_test = df[df['split'] == 'test'].copy()
            print(f"\n📊 Split 過濾:")
            print(f"  總圖像數: {len(df)} 張")
            print(f"  Test Split: {len(df_test)} 張")
            df = df_test
        
        # 顯示退化類型統計
        print("\n退化類型統計:")
        for mode in df['mode'].unique():
            count = len(df[df['mode'] == mode])
            display_name = self.config.DEGRADATION_TYPE_MAP.get(mode, mode)
            print(f"  {display_name}: {count} 張")
        
        return df
    
    def get_path_from_row(self, row, base_key: str) -> Optional[str]:
        """從 row 中獲取路徑 (嘗試多種列名變體)"""
        variants = [
            base_key,
            f"{base_key}(P)", f"{base_key} (P)",
            f"{base_key}(Mac)", f"{base_key} (Mac)",
        ]
        for col in variants:
            if col in row and pd.notna(row[col]):
                path = resolve_path(str(row[col]).strip())
                if Path(path).exists():
                    return path
        return None
    
    def save_checkpoint(self, all_results: List[Dict], processed_images: set, current_idx: int):
        """保存 checkpoint"""
        checkpoint_file = self.config.OUTPUT_DIR / "checkpoint.json"
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint_used': str(self.checkpoint_path),
                'processed_count': len(processed_images),
                'last_index': current_idx,
                'processed_images': list(processed_images),
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # 同時保存 CSV
            if all_results:
                df = pd.DataFrame(all_results)
                csv_path = self.config.LEADERBOARD_DIR / "lora_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"\n⚠️  Checkpoint 保存失敗: {e}")
    
    def load_checkpoint(self) -> tuple:
        """載入 checkpoint"""
        checkpoint_file = self.config.OUTPUT_DIR / "checkpoint.json"
        csv_path = self.config.LEADERBOARD_DIR / "lora_results.csv"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                processed_images = set(data.get('processed_images', []))
                
                # 載入已有結果
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    all_results = df.to_dict('records')
                else:
                    all_results = []
                
                print(f"\n✅ 已載入 Checkpoint:")
                print(f"  已處理: {len(processed_images)} 張")
                return all_results, processed_images
            except:
                pass
        
        return [], set()
    
    def evaluate(self, max_samples: Optional[int] = None):
        """執行評估"""
        print("\n" + "=" * 80)
        print("🎯 開始 LoRA 模型評估")
        print("=" * 80)
        
        if self.pairs_df.empty:
            print("❌ 沒有配對信息")
            return
        
        pairs_to_evaluate = self.pairs_df
        if max_samples:
            pairs_to_evaluate = pairs_to_evaluate.head(max_samples)
            print(f"\n⚠️  限制評估樣本數: {max_samples}")
        
        total_images = len(pairs_to_evaluate)
        
        # LPIPS 抽樣
        lpips_sample_size = min(self.config.LPIPS_SAMPLE_SIZE, total_images)
        lpips_sample_indices = set(random.sample(range(total_images), lpips_sample_size))
        
        print(f"\n💡 評估策略:")
        print(f"  ✅ 評估圖像: {total_images} 張 (Test Split)")
        print(f"  ✅ PSNR/SSIM: 所有圖像")
        print(f"  ✅ LPIPS: 隨機抽樣 {lpips_sample_size} 張")
        print(f"  ✅ 設備: {self.device}")
        print(f"  💾 Checkpoint: 每 {self.config.CHECKPOINT_INTERVAL} 張保存")
        
        # 載入 checkpoint
        all_results, processed_images = self.load_checkpoint()
        
        success_count = len(processed_images)
        failed_count = 0
        
        # Ctrl+C 處理：立即保存 checkpoint 再退出
        import signal
        def _handle_interrupt(signum, frame):
            print(f"\n\n⚠️  評估中斷！正在保存 checkpoint...")
            self.save_checkpoint(all_results, processed_images, -1)
            print(f"  ✅ 已保存進度（{len(processed_images)} 張）")
            print(f"  重新運行相同命令即可從此繼續")
            sys.exit(0)
        signal.signal(signal.SIGINT, _handle_interrupt)
        
        print(f"\n📊 處理圖像...")
        
        for enum_idx, (idx, row) in enumerate(tqdm(pairs_to_evaluate.iterrows(), 
                                                     total=total_images,
                                                     desc="LoRA 評估")):
            degraded_path = self.get_path_from_row(row, 'degraded_path')
            target_path = self.get_path_from_row(row, 'target_path')
            
            if not degraded_path or not target_path:
                continue
            
            image_name = Path(degraded_path).name
            
            # 跳過已處理
            if image_name in processed_images:
                continue
            
            deg_type = row['mode']
            
            try:
                # 讀取圖像
                degraded = cv2.imread(degraded_path)
                original = cv2.imread(target_path)
                
                if degraded is None or original is None:
                    continue
                
                # 確保尺寸一致
                if degraded.shape != original.shape:
                    degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
                
                # 增強（使用 tile 處理，根據設備自動調整）
                tile_size = self.config.get_tile_size(self.device)
                enhanced = enhance_image(self.model, degraded, self.device, 
                                         tile_size=tile_size, 
                                         tile_pad=self.config.TILE_PAD)
                
                # 確保增強後尺寸與原圖一致
                if enhanced.shape != original.shape:
                    enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
                
                # 計算指標
                calculate_lpips = enum_idx in lpips_sample_indices
                metrics = self.calculator.evaluate_with_reference(
                    enhanced, original, calculate_lpips=calculate_lpips
                )
                
                # 保存增強圖像
                if self.config.SAVE_ENHANCED_IMAGES:
                    enhanced_filename = f"enhanced_{Path(degraded_path).stem}.{self.config.ENHANCED_IMAGE_FORMAT}"
                    enhanced_path = self.config.ENHANCED_DIR / enhanced_filename
                    cv2.imwrite(str(enhanced_path), enhanced)
                
                # 記錄結果 (與 v2 相同格式)
                result = {
                    'image_name': image_name,
                    'degradation_type': self.config.DEGRADATION_TYPE_MAP.get(deg_type, deg_type),
                    'degradation_mode': deg_type,
                    'method': 'lora_finetuned',
                    'split': row.get('split', 'test'),
                    **metrics
                }
                all_results.append(result)
                processed_images.add(image_name)
                success_count += 1
                
            except Exception as e:
                error_msg = str(e).lower()
                # 記錄錯誤類型
                if 'convolution_overrideable' in error_msg or 'not implemented' in error_msg:
                    print(f"\n⚠️  MPS 限制（跳過）: {image_name}")
                elif 'out of memory' in error_msg:
                    print(f"\n⚠️  記憶體不足（跳過）: {image_name}")
                    # 清理記憶體
                    if self.device == 'mps':
                        torch.mps.empty_cache()
                    elif self.device == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    print(f"\n⚠️  處理失敗: {image_name} - {str(e)[:50]}")
                
                failed_count += 1
                processed_images.add(image_name)
                continue
            
            finally:
                # 清理內存
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
            
            # 定期保存
            if (success_count + failed_count) % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(all_results, processed_images, enum_idx)
                print(f"\n💾 Checkpoint #{(success_count + failed_count) // self.config.CHECKPOINT_INTERVAL}")
        
        # 最終保存
        self.save_checkpoint(all_results, processed_images, total_images)
        
        if not all_results:
            print("\n❌ 沒有生成任何結果")
            return
        
        # 生成 CSV
        csv_path = self.config.LEADERBOARD_DIR / "lora_results.csv"
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n📊 評估完成:")
        print(f"  ✅ 成功: {success_count} 張")
        print(f"  ❌ 失敗: {failed_count} 張")
        print(f"  📝 結果: {csv_path}")
        
        # 生成排行榜
        self.generate_leaderboards(df_results)
        
        print("\n✅ LoRA 評估完成！")
    
    def generate_leaderboards(self, df: pd.DataFrame):
        """生成排行榜 (與 v2 相同格式)"""
        print("\n📊 生成排行榜...")
        
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            # 聚合指標
            agg_dict = {
                'var_laplacian': 'mean',
                'tenengrad': 'mean',
                'blur_extent': 'mean',
                'luminance': 'mean',
                'contrast': 'mean',
                'entropy': 'mean',
                'gradient_mean': 'mean',
                'psnr': 'mean',
                'ssim': 'mean',
            }
            
            if 'lpips' in df_type.columns:
                agg_dict['lpips'] = 'mean'
            
            leaderboard = df_type.groupby('method').agg(agg_dict).round(4)
            
            # 計算得分 (與 v2 相同公式)
            # Reference score
            psnr_normalized = (leaderboard['psnr'] - 20) / 20
            psnr_normalized = psnr_normalized.clip(0, 1)
            leaderboard['reference_score'] = (psnr_normalized + leaderboard['ssim']) / 2
            
            # Sharpness score
            leaderboard['sharpness_score'] = (
                leaderboard['var_laplacian'] / leaderboard['var_laplacian'].max() +
                leaderboard['tenengrad'] / leaderboard['tenengrad'].max()
            ) / 2
            
            # Blur score
            leaderboard['blur_score'] = 1 - (leaderboard['blur_extent'] / leaderboard['blur_extent'].max())
            
            # Overall score
            leaderboard['overall_score'] = (
                0.6 * leaderboard['reference_score'] +
                0.2 * leaderboard['sharpness_score'] +
                0.2 * leaderboard['blur_score']
            )
            
            # 排序
            leaderboard = leaderboard.sort_values('overall_score', ascending=False)
            
            # 保存
            deg_type_safe = deg_type.replace(' ', '_').replace('-', '_').lower()
            output_path = self.config.LEADERBOARD_DIR / f"leaderboard_{deg_type_safe}.csv"
            leaderboard.to_csv(output_path, encoding='utf-8-sig')
            
            print(f"  ✓ {deg_type}: {output_path.name}")
            for method, row in leaderboard.iterrows():
                lpips_str = f", LPIPS: {row['lpips']:.4f}" if 'lpips' in row and not pd.isna(row['lpips']) else ""
                print(f"    PSNR: {row['psnr']:.2f} dB, SSIM: {row['ssim']:.4f}{lpips_str}, Overall: {row['overall_score']:.4f}")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='評估 LoRA 微調的 Real-ESRGAN 模型')
    parser.add_argument('--checkpoint', type=str, default='./lora_checkpoint_r32.pth',
                        help='LoRA checkpoint 路徑 (預設: ./lora_checkpoint_r32.pth)')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Base model 路徑 (預設: 自動解析)')
    parser.add_argument('--pairs_csv', type=str, default=None,
                        help='pairs.csv 路徑 (預設: 自動解析)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄 (預設: D:/lora_r32_eval_results 或 P:/ 或 /Volumes/Extreme SSD/)')
    parser.add_argument('--rank', type=int, default=32, help='LoRA rank (預設: 32)')
    parser.add_argument('--alpha', type=float, default=1.0, help='LoRA alpha')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='限制評估樣本數 (用於快速測試)')
    parser.add_argument('--no_save_images', action='store_true',
                        help='不保存增強後的圖像')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚀 LoRA 微調模型評估")
    print("=" * 80)
    
    # 配置
    config = LoRAEvalConfig()
    config.LORA_RANK = args.rank
    config.LORA_ALPHA = args.alpha
    
    # 初始化輸出目錄（跨平台: D:/ -> P:/ -> /Volumes/Extreme SSD/）
    config.init_output_dirs(args.output_dir)
    
    config.SAVE_ENHANCED_IMAGES = not args.no_save_images
    
    if args.base_model:
        config.BASE_MODEL_PATH = args.base_model
    if args.pairs_csv:
        config.PAIRS_CSV = args.pairs_csv
    
    print(f"\n📋 配置:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Base model: {config.BASE_MODEL_PATH}")
    print(f"  pairs.csv: {config.PAIRS_CSV}")
    print(f"  📂 Output: {config.OUTPUT_DIR}")
    print(f"  LoRA Rank: {config.LORA_RANK}")
    print(f"  設備: {DEVICE}")
    
    # 創建評估器並執行
    evaluator = LoRAEvaluator(args.checkpoint, config)
    evaluator.evaluate(max_samples=args.max_samples)
    
    print(f"\n📁 結果已保存到: {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
