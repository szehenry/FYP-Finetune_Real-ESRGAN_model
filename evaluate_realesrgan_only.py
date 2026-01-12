#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 單獨評估腳本
========================================

目的：只評估 Real-ESRGAN（未微調版本）作為基準
     可以與已有的傳統方法結果合併

特點：
1. ✅ 使用與 HenryCC 版本相同的配置
2. ✅ 僅評估 Real-ESRGAN（節省時間）
3. ✅ 輸出格式與傳統方法一致（便於合併）
4. ✅ GPU 記憶體管理優化
5. ✅ 支持 LPIPS 隨機抽樣

作者：FYP Project
日期：2025-12
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import random

# 設置 UTF-8 輸出（修復 Windows 編碼問題）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 設置隨機種子（確保可重現）
random.seed(42)
np.random.seed(42)

# 添加 Real-ESRGAN 到路徑
REALESRGAN_PATH = Path(r"D:\Real-ESRGAN")
if REALESRGAN_PATH.exists():
    sys.path.insert(0, str(REALESRGAN_PATH))

# 檢查 Real-ESRGAN 是否可用
try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
    print("✓ Real-ESRGAN 導入成功")
except ImportError as e:
    print(f"❌ Real-ESRGAN 未安裝或導入失敗: {e}")
    print("請參考 setup_realesrgan_env.md 設置環境")
    sys.exit(1)

# 導入基準評估模組
try:
    from baseline_evaluation import MetricsCalculator, Config as BaseConfig
    print("✓ 成功導入基準方法模組")
except ImportError as e:
    print(f"❌ 無法導入 baseline_evaluation 模組: {e}")
    print("請確保 baseline_evaluation.py 在相同目錄下")
    sys.exit(1)


# ==================== 配置區 ====================

class RealESRGANOnlyConfig(BaseConfig):
    """Real-ESRGAN 單獨評估配置"""
    
    # 路徑配置（與 HenryCC 版本一致）
    PAIRS_CSV = Path(r"D:\degraded_full_dataset\pairs.csv")
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset\degraded")
    ORIGINAL_DIR = Path(r"D:\FYP_Images")
    
    # Real-ESRGAN 專用路徑
    REALESRGAN_ROOT = Path(r"D:\Real-ESRGAN")
    MODEL_PATH = REALESRGAN_ROOT / "experiments" / "pretrained_models" / "RealESRGAN_x4plus.pth"
    
    # 輸出目錄（新建，避免覆蓋舊結果）
    OUTPUT_DIR = Path(r"D:\baseline_results_realesrgan_only")
    RESULTS_DIR = OUTPUT_DIR / "enhanced_images"
    COMPARISON_DIR = OUTPUT_DIR / "comparison_samples"
    LEADERBOARD_DIR = OUTPUT_DIR / "leaderboards"
    FAILURE_DIR = OUTPUT_DIR / "failure_cases"
    
    # 退化類型映射
    DEGRADATION_TYPE_MAP = {
        'global_blur': 'Drone Motion Blur',
        'object_blur': 'Object Motion Blur',
        'low_light': 'Low-light'
    }
    
    # LPIPS 抽樣配置（與 HenryCC 一致）
    LPIPS_SAMPLE_SIZE = 700
    LPIPS_ENABLED = True
    
    # Real-ESRGAN 參數
    TILE = 512  # 平衡內存和速度（512 是安全值，適合 RTX 3060 6GB/12GB）
    TILE_PAD = 10
    PRE_PAD = 0
    FP32 = True  # 使用 FP32（更穩定，避免內存碎片）
    OUTSCALE = 1  # 保持原大小
    
    # Checkpoint 配置
    CHECKPOINT_ENABLED = True  # 啟用 checkpoint 恢復
    CHECKPOINT_INTERVAL = 50  # 每 50 張保存一次
    CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
    
    # 設備
    try:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        DEVICE = 'cpu'


# ==================== Real-ESRGAN 處理器 ====================

class RealESRGANProcessor:
    """Real-ESRGAN 圖像處理器"""
    
    def __init__(self, model_path: Path, config: RealESRGANOnlyConfig):
        """
        初始化 Real-ESRGAN
        
        Args:
            model_path: 模型權重路徑
            config: 配置對象
        """
        self.config = config
        self.device = config.DEVICE
        self.model_path = model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
        
        print(f"\n📦 載入 Real-ESRGAN 模型...")
        print(f"  模型: {model_path.name}")
        print(f"  設備: {self.device}")
        print(f"  Tile 大小: {config.TILE}")
        
        # 初始化模型
        self.model = self._load_model()
    
    def _load_model(self):
        """載入 Real-ESRGAN 模型"""
        # 定義網絡架構（RealESRGAN_x4plus）
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
            tile=self.config.TILE,
            tile_pad=self.config.TILE_PAD,
            pre_pad=self.config.PRE_PAD,
            half=not self.config.FP32,  # FP16 加速
            device=self.device
        )
        
        print("✓ 模型載入成功\n")
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
                outscale=self.config.OUTSCALE
            )
            
            # 確保大小一致（如果 Real-ESRGAN 改變了大小）
            if output.shape[:2] != img.shape[:2]:
                output = cv2.resize(output, (img.shape[1], img.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
            
            return output
        
        except Exception as e:
            print(f"  ⚠️  Real-ESRGAN 處理失敗: {e}")
            return img  # 返回原圖


# ==================== 主評估器 ====================

class RealESRGANOnlyEvaluator:
    """Real-ESRGAN 單獨評估器"""
    
    def __init__(self):
        self.config = RealESRGANOnlyConfig()
        self.calculator = MetricsCalculator()
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Real-ESRGAN 處理器
        print("=" * 80)
        print("🚀 初始化 Real-ESRGAN 處理器")
        print("=" * 80)
        self.realesrgan = RealESRGANProcessor(
            self.config.MODEL_PATH,
            self.config
        )
        
        # 載入配對信息
        self.pairs_df = self.load_pairs()
    
    def load_pairs(self) -> pd.DataFrame:
        """載入 pairs.csv"""
        if not self.config.PAIRS_CSV.exists():
            print(f"❌ 配對文件不存在: {self.config.PAIRS_CSV}")
            return pd.DataFrame()
        
        print(f"📁 載入配對信息: {self.config.PAIRS_CSV}")
        df = pd.read_csv(self.config.PAIRS_CSV)
        print(f"✓ 載入 {len(df)} 對配對（包含所有增強版本，與 baseline 一致）")
        
        # 顯示退化類型統計
        print("\n退化類型統計:")
        for mode in df['mode'].unique():
            count = len(df[df['mode'] == mode])
            display_name = self.config.DEGRADATION_TYPE_MAP.get(mode, mode)
            print(f"  {display_name}: {count} 張")
        
        return df
    
    def evaluate_realesrgan(self):
        """評估 Real-ESRGAN"""
        print("\n" + "=" * 80)
        print("🎯 開始 Real-ESRGAN 評估（有參考指標 + LPIPS 隨機抽樣）")
        print("=" * 80)
        
        if self.pairs_df.empty:
            print("❌ 沒有配對信息，無法進行評估")
            return
        
        # 隨機選擇要計算 LPIPS 的圖像索引
        total_images = len(self.pairs_df)
        sample_size = min(self.config.LPIPS_SAMPLE_SIZE, total_images)
        lpips_sample_indices = set(random.sample(range(total_images), sample_size))
        
        print(f"\n💡 評估策略:")
        print(f"  ✅ PSNR/SSIM: 所有 {total_images} 張圖像")
        print(f"  ✅ LPIPS: 隨機抽樣 {sample_size} 張 ({sample_size/total_images*100:.1f}%)")
        print(f"  🔧 記憶體管理: 每張圖像後清理 GPU 快取")
        print(f"  ⚡ 使用設備: {self.config.DEVICE}")
        print(f"\n📈 預計時間: ~23-28 小時（與 HenryCC baseline 一致）")
        
        # 初始化結果存儲
        all_results = []
        
        # 處理每對圖像
        print(f"\n📊 處理 {len(self.pairs_df)} 對圖像...")
        for enum_idx, (idx, row) in enumerate(tqdm(self.pairs_df.iterrows(), 
                                                    total=len(self.pairs_df), 
                                                    desc="Real-ESRGAN 評估")):
            degraded_path = Path(row['degraded_path'])
            original_path = Path(row['target_path'])
            deg_type = row['mode']
            
            # 檢查文件是否存在
            if not degraded_path.exists():
                print(f"  ⚠️  退化圖像不存在: {degraded_path.name}")
                continue
            
            if not original_path.exists():
                print(f"  ⚠️  原始圖像不存在: {original_path.name}")
                continue
            
            # 讀取圖像
            degraded = cv2.imread(str(degraded_path))
            original = cv2.imread(str(original_path))
            
            if degraded is None or original is None:
                continue
            
            # 確保尺寸一致
            if degraded.shape != original.shape:
                degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
            
            # 判斷是否對這張圖計算 LPIPS（使用 enum_idx 確保從0開始的連續索引）
            calculate_lpips_for_this_image = enum_idx in lpips_sample_indices
            
            try:
                # 應用 Real-ESRGAN
                enhanced = self.realesrgan.enhance(degraded)
                
                # 計算有參考指標（僅對抽樣的圖像計算 LPIPS）
                metrics = self.calculator.evaluate_with_reference(
                    enhanced, original, 
                    calculate_lpips=calculate_lpips_for_this_image
                )
                
                # 記錄結果
                result_entry = {
                    'image_name': degraded_path.name,
                    'degradation_type': self.config.DEGRADATION_TYPE_MAP.get(deg_type, deg_type),
                    'degradation_mode': deg_type,
                    'method': 'realesrgan',  # 方法名
                    'split': row['split'],
                    **metrics
                }
                all_results.append(result_entry)
                
                # 🔧 每張圖像後立即清理記憶體（與 HenryCC 版本一致）
                del degraded, original, enhanced
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"  ⚠️  Real-ESRGAN 處理 {degraded_path.name} 失敗: {e}")
                continue
        
        if not all_results:
            print("❌ 沒有生成任何結果")
            return
        
        # 保存結果到 DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存完整結果
        full_results_path = self.config.LEADERBOARD_DIR / "realesrgan_results.csv"
        df.to_csv(full_results_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Real-ESRGAN 結果已保存: {full_results_path}")
        
        # 生成統計摘要
        self.generate_statistics_summary(df)
        
        print("\n" + "=" * 80)
        print("✅ Real-ESRGAN 評估完成！")
        print("=" * 80)
        print(f"\n📁 結果已保存至: {self.config.OUTPUT_DIR}")
        print(f"\n💡 下一步：運行 merge_baseline_results.py 合併所有結果")
    
    def generate_statistics_summary(self, df: pd.DataFrame):
        """生成統計摘要"""
        print("\n📈 生成統計摘要...")
        
        summary = {
            'method': 'Real-ESRGAN (unfinetuned)',
            'total_images': len(df['image_name'].unique()),
            'total_evaluations': len(df),
            'degradation_types': {},
            'overall_metrics': {}
        }
        
        # 按退化類型統計
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            type_metrics = {
                'count': len(df_type['image_name'].unique()),
            }
            
            # 計算平均指標
            if 'psnr' in df_type.columns:
                type_metrics['avg_psnr'] = float(df_type['psnr'].mean())
            if 'ssim' in df_type.columns:
                type_metrics['avg_ssim'] = float(df_type['ssim'].mean())
            if 'lpips' in df_type.columns:
                lpips_values = df_type['lpips'].dropna()
                if len(lpips_values) > 0:
                    type_metrics['avg_lpips'] = float(lpips_values.mean())
            
            summary['degradation_types'][deg_type] = type_metrics
        
        # 總體指標
        if 'psnr' in df.columns:
            summary['overall_metrics']['psnr'] = float(df['psnr'].mean())
        if 'ssim' in df.columns:
            summary['overall_metrics']['ssim'] = float(df['ssim'].mean())
        if 'lpips' in df.columns:
            lpips_values = df['lpips'].dropna()
            if len(lpips_values) > 0:
                summary['overall_metrics']['lpips'] = float(lpips_values.mean())
        
        # 保存摘要
        summary_path = self.config.LEADERBOARD_DIR / "realesrgan_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 統計摘要已保存: {summary_path.name}")
        
        # 打印關鍵指標
        print(f"\n  📊 Real-ESRGAN 性能:")
        for deg_type, metrics in summary['degradation_types'].items():
            psnr_str = f"PSNR: {metrics.get('avg_psnr', 0):.2f} dB" if 'avg_psnr' in metrics else ""
            ssim_str = f"SSIM: {metrics.get('avg_ssim', 0):.4f}" if 'avg_ssim' in metrics else ""
            print(f"    {deg_type:20s} | {psnr_str:15s} | {ssim_str}")


# ==================== 主程序 ====================

def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("🚀 Real-ESRGAN 基準評估（單獨運行）")
    print("=" * 80)
    print("\n📋 配置資訊:")
    print(f"  退化圖像目錄: {RealESRGANOnlyConfig.DEGRADED_DIR}")
    print(f"  原始圖像目錄: {RealESRGANOnlyConfig.ORIGINAL_DIR}")
    print(f"  配對文件: {RealESRGANOnlyConfig.PAIRS_CSV}")
    print(f"  模型路徑: {RealESRGANOnlyConfig.MODEL_PATH}")
    print(f"  輸出目錄: {RealESRGANOnlyConfig.OUTPUT_DIR}")
    print(f"  設備: {RealESRGANOnlyConfig.DEVICE}")
    print(f"  LPIPS 隨機抽樣: {RealESRGANOnlyConfig.LPIPS_SAMPLE_SIZE} 張")
    
    # 創建評估器
    evaluator = RealESRGANOnlyEvaluator()
    
    # 執行評估
    evaluator.evaluate_realesrgan()
    
    print("\n✅ 所有任務完成！")
    print(f"\n📁 請檢查輸出目錄: {RealESRGANOnlyConfig.OUTPUT_DIR}")
    print("\n💡 下一步：")
    print("   1. 運行 merge_baseline_results.py 合併所有結果")
    print("   2. 生成包含 9 個方法的完整對比圖")


if __name__ == "__main__":
    main()

