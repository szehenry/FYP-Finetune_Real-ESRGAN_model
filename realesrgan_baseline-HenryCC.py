#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 基準測試 (Real-ESRGAN Baseline Evaluation)
======================================================

目的：使用 Real-ESRGAN 作為強基準（strong baseline）進行比較

功能：
1. 使用 Real-ESRGAN 預訓練模型處理退化圖像
2. 計算相同的評估指標
3. 與簡單基準方法對比
4. 生成排行榜和可視化

使用前提：
- 已按照 setup_realesrgan_env.md 設置環境
- 已下載預訓練模型到 D:\Real-ESRGAN\experiments\pretrained_models\

作者：FYP Project
日期：2025-11
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加 Real-ESRGAN 到路徑
REALESRGAN_PATH = Path(r"D:\Real-ESRGAN")
if REALESRGAN_PATH.exists():
    sys.path.insert(0, str(REALESRGAN_PATH))

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Real-ESRGAN 未安裝或導入失敗: {e}")
    print("請參考 setup_realesrgan_env.md 設置環境")
    REALESRGAN_AVAILABLE = False

# 導入基準評估模組（使用前面創建的）
try:
    from baseline_evaluation import MetricsCalculator, Config as BaseConfig
except ImportError:
    print("⚠️  無法導入 baseline_evaluation 模組")
    print("請確保 baseline_evaluation.py 在相同目錄下")
    sys.exit(1)


# ==================== 配置區 ====================

class RealESRGANConfig:
    """Real-ESRGAN 配置"""
    
    # 路徑
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset")
    OUTPUT_DIR = Path(r"D:\realesrgan_results")
    REALESRGAN_ROOT = Path(r"D:\Real-ESRGAN")
    
    # 模型配置
    MODEL_PATH = REALESRGAN_ROOT / "experiments" / "pretrained_models" / "RealESRGAN_x4plus.pth"
    MODEL_NAME = "RealESRGAN_x4plus"
    
    # 備選模型
    AVAILABLE_MODELS = {
        'x4plus': 'RealESRGAN_x4plus.pth',
        'x2plus': 'RealESRGAN_x2plus.pth',
        'x4plus_anime': 'RealESRGAN_x4plus_anime_6B.pth',
    }
    
    # Real-ESRGAN 參數
    DENOISE_STRENGTH = 0.5  # 0-1, 去噪強度
    OUTSCALE = 1  # 輸出縮放（我們保持原大小）
    TILE = 400  # 分塊處理大小（避免記憶體不足）
    TILE_PAD = 10
    PRE_PAD = 0
    FP32 = False  # 是否使用 FP32（更慢但更精確）
    
    # 設備
    try:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        DEVICE = 'cpu'
    
    # 輸出
    ENHANCED_DIR = OUTPUT_DIR / "enhanced"
    COMPARISON_DIR = OUTPUT_DIR / "comparisons"
    LEADERBOARD_DIR = OUTPUT_DIR / "leaderboards"
    
    # 樣本數
    NUM_SAMPLES = 20


# ==================== Real-ESRGAN 處理器 ====================

class RealESRGANProcessor:
    """Real-ESRGAN 圖像處理器"""
    
    def __init__(self, model_path: Path, device: str = 'cuda'):
        """
        初始化 Real-ESRGAN
        
        Args:
            model_path: 模型權重路徑
            device: 'cuda' 或 'cpu'
        """
        if not REALESRGAN_AVAILABLE:
            raise RuntimeError("Real-ESRGAN 未安裝")
        
        self.device = device
        self.model_path = model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"📦 載入 Real-ESRGAN 模型: {model_path.name}")
        print(f"🖥️  設備: {device}")
        
        # 初始化模型
        self.model = self._load_model()
    
    def _load_model(self):
        """載入 Real-ESRGAN 模型"""
        # 定義網絡架構
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # 創建 Real-ESRGAN upsampler
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.model_path),
            model=model,
            tile=RealESRGANConfig.TILE,
            tile_pad=RealESRGANConfig.TILE_PAD,
            pre_pad=RealESRGANConfig.PRE_PAD,
            half=not RealESRGANConfig.FP32,
            device=self.device
        )
        
        print("✓ 模型載入成功")
        return upsampler
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        使用 Real-ESRGAN 增強圖像
        
        Args:
            img: 輸入圖像 (BGR, uint8)
        
        Returns:
            enhanced: 增強後的圖像
        """
        try:
            # Real-ESRGAN 推理
            output, _ = self.model.enhance(
                img,
                outscale=RealESRGANConfig.OUTSCALE
            )
            
            # 確保大小一致（如果 Real-ESRGAN 改變了大小）
            if output.shape[:2] != img.shape[:2]:
                output = cv2.resize(output, (img.shape[1], img.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
            
            return output
        
        except Exception as e:
            print(f"⚠️  Real-ESRGAN 處理失敗: {e}")
            return img  # 返回原圖


# ==================== 主評估器 ====================

class RealESRGANEvaluator:
    """Real-ESRGAN 評估器"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.config = RealESRGANConfig()
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Real-ESRGAN
        model_path = model_path or self.config.MODEL_PATH
        
        if not REALESRGAN_AVAILABLE:
            print("❌ Real-ESRGAN 不可用，請先設置環境")
            sys.exit(1)
        
        try:
            self.processor = RealESRGANProcessor(model_path, self.config.DEVICE)
        except Exception as e:
            print(f"❌ Real-ESRGAN 初始化失敗: {e}")
            sys.exit(1)
        
        # 初始化指標計算器
        self.calculator = MetricsCalculator()
    
    def find_degraded_images(self) -> List[Tuple[Path, str, str]]:
        """查找所有退化圖像"""
        degraded_images = []
        
        if not self.config.DEGRADED_DIR.exists():
            print(f"❌ 退化圖像目錄不存在: {self.config.DEGRADED_DIR}")
            return degraded_images
        
        # 遍歷退化類型
        degradation_types = ['drone_motion_blur', 'object_motion_blur', 'low_light']
        
        for deg_type in degradation_types:
            deg_dir = self.config.DEGRADED_DIR / deg_type
            if not deg_dir.exists():
                continue
            
            # 查找所有圖像
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                for img_path in deg_dir.glob(f'*{ext}'):
                    degraded_images.append((img_path, deg_type, img_path.name))
        
        print(f"✓ 找到 {len(degraded_images)} 張退化圖像")
        return degraded_images
    
    def process_all_images(self):
        """處理所有圖像"""
        print("\n" + "=" * 80)
        print("🚀 Real-ESRGAN 基準測試")
        print("=" * 80)
        
        # 查找圖像
        degraded_images = self.find_degraded_images()
        if not degraded_images:
            print("❌ 沒有找到退化圖像")
            return
        
        # 初始化結果
        all_results = []
        
        # 處理每張圖像
        print(f"\n📊 處理 {len(degraded_images)} 張圖像...")
        for img_path, deg_type, orig_name in tqdm(degraded_images, desc="Real-ESRGAN 推理"):
            # 讀取圖像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Real-ESRGAN 增強
            enhanced = self.processor.enhance(img)
            
            # 保存增強圖像
            output_dir = self.config.ENHANCED_DIR / deg_type
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / orig_name
            cv2.imwrite(str(output_path), enhanced)
            
            # 計算指標
            metrics = self.calculator.evaluate_no_reference(enhanced)
            
            # 記錄結果
            result_entry = {
                'image_name': orig_name,
                'degradation_type': deg_type,
                'method': 'RealESRGAN',
                **metrics
            }
            all_results.append(result_entry)
        
        # 保存結果
        df = pd.DataFrame(all_results)
        results_path = self.config.LEADERBOARD_DIR / "realesrgan_results.csv"
        df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 結果已保存: {results_path}")
        
        # 生成統計
        self.generate_statistics(df)
        
        # 生成對比樣本
        self.generate_comparisons(degraded_images[:self.config.NUM_SAMPLES])
        
        print("\n" + "=" * 80)
        print("✅ Real-ESRGAN 評估完成！")
        print("=" * 80)
    
    def generate_statistics(self, df: pd.DataFrame):
        """生成統計報告"""
        print("\n📊 統計摘要:")
        
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            print(f"\n  {deg_type}:")
            print(f"    圖像數: {len(df_type)}")
            print(f"    平均 Var Laplacian: {df_type['var_laplacian'].mean():.2f}")
            print(f"    平均 Tenengrad: {df_type['tenengrad'].mean():.2f}")
            print(f"    平均 Blur Extent: {df_type['blur_extent'].mean():.4f}")
            
            if 'niqe' in df_type.columns:
                print(f"    平均 NIQE: {df_type['niqe'].mean():.4f}")
            if 'brisque' in df_type.columns:
                print(f"    平均 BRISQUE: {df_type['brisque'].mean():.4f}")
    
    def generate_comparisons(self, sample_images: List[Tuple[Path, str, str]]):
        """生成對比圖"""
        print(f"\n🖼️  生成對比圖（前 {len(sample_images)} 張）...")
        
        import matplotlib.pyplot as plt
        
        for img_path, deg_type, orig_name in tqdm(sample_images, desc="生成對比"):
            # 讀取原圖
            degraded = cv2.imread(str(img_path))
            if degraded is None:
                continue
            
            # 增強
            enhanced = self.processor.enhance(degraded)
            
            # 創建對比圖
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Degraded
            axes[0].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Degraded ({deg_type})', fontsize=12)
            axes[0].axis('off')
            
            # Enhanced
            axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Real-ESRGAN Enhanced', fontsize=12)
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # 保存
            output_name = f"{Path(orig_name).stem}_{deg_type}_realesrgan.png"
            output_path = self.config.COMPARISON_DIR / output_name
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 對比圖已保存至: {self.config.COMPARISON_DIR}")


# ==================== 主程序 ====================

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Real-ESRGAN 基準測試')
    parser.add_argument('--model', type=str, default='x4plus',
                       choices=['x4plus', 'x2plus', 'x4plus_anime'],
                       help='選擇模型')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='選擇設備')
    parser.add_argument('--test', action='store_true',
                       help='測試模式（僅處理幾張圖像）')
    
    args = parser.parse_args()
    
    # 設置設備
    if args.device == 'auto':
        device = RealESRGANConfig.DEVICE
    else:
        device = args.device
    
    RealESRGANConfig.DEVICE = device
    
    # 設置模型路徑
    model_name = RealESRGANConfig.AVAILABLE_MODELS[args.model]
    model_path = RealESRGANConfig.REALESRGAN_ROOT / "experiments" / "pretrained_models" / model_name
    
    # 檢查模型是否存在
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n請下載模型:")
        print(f"  模型: {model_name}")
        print(f"  放置位置: {model_path.parent}")
        print("\n下載連結: https://github.com/xinntao/Real-ESRGAN/releases")
        return
    
    print(f"\n⚙️  配置:")
    print(f"  模型: {args.model}")
    print(f"  設備: {device}")
    print(f"  測試模式: {'是' if args.test else '否'}")
    
    # 創建評估器
    evaluator = RealESRGANEvaluator(model_path)
    
    # 如果是測試模式，限制圖像數量
    if args.test:
        print("\n⚠️  測試模式：僅處理前 5 張圖像")
        RealESRGANConfig.NUM_SAMPLES = 5
        # 暫時修改 find_degraded_images 以限制數量
        original_find = evaluator.find_degraded_images
        evaluator.find_degraded_images = lambda: original_find()[:5]
    
    # 執行處理
    evaluator.process_all_images()
    
    print(f"\n📁 結果目錄: {RealESRGANConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

