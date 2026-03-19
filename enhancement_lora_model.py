#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA-finetuned Real-ESRGAN 單圖增強腳本
========================================

功能：使用訓練好的 LoRA 模型增強單張圖像，無需評估指標。

用法:
  source venv_lora_mac/bin/activate   # Mac
  # 或
  .\venv_lora_win\Scripts\activate         # Windows

  python enhancement_lora_model.py

  運行後會提示輸入圖像路徑，增強結果保存到 enhanced_image_output 資料夾。

作者：FYP Project
日期：2025-01
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import re
import gc
from tqdm import tqdm

import torch
import torch.nn as nn

# 檢查設備
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# 輸出目錄（相對於腳本所在目錄）
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "./enhanced_image_output"

# 跨平台路徑
BASE_PATHS = ['D:/', 'P:/', '/Volumes/Extreme SSD/']


def resolve_path(path_str: str) -> str:
    """跨平台路徑解析"""
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.exists():
        return str(p)
    m = re.match(r'^([A-Za-z]):[/\\](.*)$', str(path_str))
    if m:
        rel = m.group(2).replace('\\', '/')
        for base in BASE_PATHS:
            test = Path(base) / rel
            if test.exists():
                return str(test)
    return path_str


# ==================== 模型定義 ====================

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


# ==================== 模型載入與增強 ====================

def load_lora_model(checkpoint_path: Path, base_model_path: Path, device: str, rank: int = 32, alpha: float = 1.0):
    """載入 LoRA 微調模型"""
    base_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    if base_model_path.exists():
        ckpt = torch.load(str(base_model_path), map_location='cpu', weights_only=True)
        base_model.load_state_dict(ckpt.get('params_ema', ckpt), strict=False)
    model = RRDBWithLoRA(base_model, rank=rank, alpha=alpha)
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'], strict=False)
    model.to(device)
    model.eval()
    return model


def enhance_image(model, img: np.ndarray, device: str, tile_size: int = 256, tile_pad: int = 32,
                  show_progress: bool = True) -> np.ndarray:
    """使用 LoRA 模型增強圖像（與 evaluate_lora_model.py 相同邏輯）"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    original_h, original_w = img.shape[:2]

    if device == 'mps':
        tile_size = 128
        new_h = ((h + 63) // 64) * 64
        new_w = ((w + 63) // 64) * 64
        if new_h != h or new_w != w:
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w

    use_tile = (device == 'mps') or (h > 512 or w > 512)
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
        if output_bgr.shape[0] != original_h or output_bgr.shape[1] != original_w:
            output_bgr = cv2.resize(output_bgr, (original_w, original_h), interpolation=cv2.INTER_AREA)
        return output_bgr

    output_h, output_w = h * scale, w * scale
    output_acc = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight_acc = np.zeros((output_h, output_w, 1), dtype=np.float32)
    step = tile_size

    y_positions = list(range(0, h - tile_size + 1, step))
    if y_positions and y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    x_positions = list(range(0, w - tile_size + 1, step))
    if x_positions and x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    tile_pairs = [(y, x) for y in y_positions for x in x_positions]
    total_tiles = len(tile_pairs)
    if show_progress and total_tiles > 1:
        print(f"  處理 {total_tiles} 個 tiles (大圖需較長時間)...")

    iterator = tqdm(tile_pairs, desc="  Tiles", unit="tile", disable=not show_progress or total_tiles <= 1)

    for y_start, x_start in iterator:
            y_start_pad = max(0, y_start - tile_pad)
            y_end_pad = min(h, y_start + tile_size + tile_pad)
            x_start_pad = max(0, x_start - tile_pad)
            x_end_pad = min(w, x_start + tile_size + tile_pad)

            tile = img_rgb[y_start_pad:y_end_pad, x_start_pad:x_end_pad].copy()
            th, tw = tile.shape[:2]

            if device == 'mps':
                new_th = ((th + 63) // 64) * 64
                new_tw = ((tw + 63) // 64) * 64
                if new_th != th or new_tw != tw:
                    tile = cv2.resize(tile, (new_tw, new_th), interpolation=cv2.INTER_CUBIC)

            tile_tensor = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tile_output = process_tensor(tile_tensor)
            tile_output_np = tile_output.squeeze(0).permute(1, 2, 0).cpu().numpy()

            expected_h = th * scale
            expected_w = tw * scale
            if tile_output_np.shape[0] != expected_h or tile_output_np.shape[1] != expected_w:
                tile_output_np = cv2.resize(tile_output_np, (expected_w, expected_h), interpolation=cv2.INTER_CUBIC)

            fade_size = tile_pad * scale
            tile_weight = np.ones((expected_h, expected_w, 1), dtype=np.float32)
            for i in range(min(fade_size, expected_h // 2)):
                factor = (i + 1) / (fade_size + 1)
                tile_weight[i, :, 0] *= factor
                tile_weight[expected_h - 1 - i, :, 0] *= factor
            for i in range(min(fade_size, expected_w // 2)):
                factor = (i + 1) / (fade_size + 1)
                tile_weight[:, i, 0] *= factor
                tile_weight[:, expected_w - 1 - i, 0] *= factor

            out_y_start = y_start_pad * scale
            out_y_end = y_end_pad * scale
            out_x_start = x_start_pad * scale
            out_x_end = x_end_pad * scale

            output_acc[out_y_start:out_y_end, out_x_start:out_x_end] += tile_output_np * tile_weight
            weight_acc[out_y_start:out_y_end, out_x_start:out_x_end] += tile_weight

            del tile_tensor, tile_output
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()

    weight_acc = np.maximum(weight_acc, 1e-8)
    output_np = output_acc / weight_acc
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    # 縮回原圖尺寸（與 evaluate 一致，避免輸出過大）
    if output_bgr.shape[0] != original_h or output_bgr.shape[1] != original_w:
        output_bgr = cv2.resize(output_bgr, (original_w, original_h), interpolation=cv2.INTER_AREA)
    return output_bgr


# ==================== 主程序 ====================

def main():
    print("\n" + "=" * 60)
    print("🖼️  LoRA 單圖增強")
    print("=" * 60)

    checkpoint_path = SCRIPT_DIR / "lora_checkpoint_r32.pth"
    base_model_path = Path(resolve_path("D:/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"))
    if not base_model_path.exists():
        base_model_path = Path(resolve_path("P:/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"))
    if not base_model_path.exists():
        base_model_path = Path(resolve_path("/Volumes/Extreme SSD/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"))

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)
    if not base_model_path.exists():
        print(f"❌ Base model 不存在: {base_model_path}")
        sys.exit(1)

    print(f"\n載入模型中... (設備: {DEVICE})")
    model = load_lora_model(checkpoint_path, base_model_path, DEVICE, rank=32, alpha=1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n輸出目錄: {OUTPUT_DIR}")

    while True:
        print("\n" + "-" * 40)
        path_input = input("請輸入圖像路徑 (或輸入 q 退出): ").strip()
        if path_input.lower() in ('q', 'quit', 'exit'):
            print("再見！")
            break

        path_input = path_input.strip('"').strip("'")
        img_path = Path(resolve_path(path_input))
        if not img_path.exists():
            print(f"❌ 找不到檔案: {path_input}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ 無法讀取圖像: {img_path}")
            continue

        h_in, w_in = img.shape[:2]
        print(f"處理中: {img_path.name} ({w_in}×{h_in})...")
        if h_in > 2000 or w_in > 2000:
            print(f"  ⚠️  大圖 ({w_in}×{h_in}) 需較長時間，請耐心等待...")
        try:
            enhanced = enhance_image(model, img, DEVICE, tile_size=256 if DEVICE == 'cuda' else 128, tile_pad=32)
            out_name = f"enhanced_{img_path.stem}.png"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), enhanced)
            print(f"✅ 已保存: {out_path}")
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
        finally:
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            elif DEVICE == 'mps':
                torch.mps.empty_cache()


if __name__ == '__main__':
    main()
