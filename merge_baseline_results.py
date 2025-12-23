#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合併基準結果並生成完整對比圖
========================================

目的：
1. 合併傳統方法結果（D:\baseline_results_old）和 Real-ESRGAN 結果
2. 生成包含所有 9 個方法的完整排行榜
3. 重新生成對比樣本（50 張，包含所有方法）

使用流程：
1. 先運行 baseline_evaluation_with_gt-HenryCC.py（或使用已有結果）
2. 運行 evaluate_realesrgan_only.py
3. 運行本腳本合併結果

作者：FYP Project
日期：2025-12
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

# 設置 UTF-8 輸出（修復 Windows 編碼問題）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 導入基準方法（用於生成對比圖）
try:
    from baseline_evaluation import BaselineMethods
    print("✓ 成功導入基準方法模組")
except ImportError as e:
    print(f"❌ 無法導入 baseline_evaluation 模組: {e}")
    sys.exit(1)

# 導入 Real-ESRGAN（用於生成對比圖）
REALESRGAN_PATH = Path(r"D:\Real-ESRGAN")
if REALESRGAN_PATH.exists():
    sys.path.insert(0, str(REALESRGAN_PATH))

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
    print("✓ Real-ESRGAN 導入成功")
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("⚠️  Real-ESRGAN 未安裝，將跳過 Real-ESRGAN 對比圖生成")


# ==================== 配置區 ====================

class MergeConfig:
    """合併配置"""
    
    # 輸入路徑
    OLD_RESULTS_DIR = Path(r"D:\baseline_results_old\leaderboards")
    REALESRGAN_RESULTS_DIR = Path(r"D:\baseline_results_realesrgan_only\leaderboards")
    
    # 配對文件和圖像路徑
    PAIRS_CSV = Path(r"D:\degraded_full_dataset\pairs.csv")
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset\degraded")
    ORIGINAL_DIR = Path(r"D:\FYP_Images")
    
    # Real-ESRGAN 配置
    REALESRGAN_ROOT = Path(r"D:\Real-ESRGAN")
    MODEL_PATH = REALESRGAN_ROOT / "experiments" / "pretrained_models" / "RealESRGAN_x4plus.pth"
    TILE = 400
    TILE_PAD = 10
    PRE_PAD = 0
    FP32 = False
    OUTSCALE = 1
    
    try:
        DEVICE = 'cuda' if 'torch' in dir() and torch.cuda.is_available() else 'cpu'
    except:
        DEVICE = 'cpu'
    
    # 輸出路徑
    OUTPUT_DIR = Path(r"D:\baseline_results_merged")
    LEADERBOARD_DIR = OUTPUT_DIR / "leaderboards"
    COMPARISON_DIR = OUTPUT_DIR / "comparison_samples"
    FAILURE_DIR = OUTPUT_DIR / "failure_cases"
    
    # 對比樣本數量（增加到 50）
    NUM_SAMPLES = 50
    NUM_FAILURE_SAMPLES = 10
    
    # 退化類型映射
    DEGRADATION_TYPE_MAP = {
        'global_blur': 'Drone Motion Blur',
        'object_blur': 'Object Motion Blur',
        'low_light': 'Low-light'
    }
    
    # 方法順序（用於對比圖排列）
    METHOD_ORDER = [
        'Original (GT)',
        'Degraded',
        'identity',
        'bicubic',
        'gaussian',
        'sharpen',
        'unsharp',
        'combined',
        'realesrgan'
    ]
    
    # 方法顯示名稱
    METHOD_DISPLAY_NAMES = {
        'identity': 'Identity',
        'bicubic': 'Bicubic',
        'gaussian': 'Gaussian',
        'sharpen': 'Sharpen',
        'unsharp': 'Unsharp Mask',
        'combined': 'Combined',
        'realesrgan': 'Real-ESRGAN'
    }


# ==================== 結果合併器 ====================

class BaselineResultsMerger:
    """基準結果合併器"""
    
    def __init__(self):
        self.config = MergeConfig()
        self.methods = BaselineMethods()
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 基準方法字典
        self.baseline_methods = {
            'identity': self.methods.identity,
            'bicubic': self.methods.bicubic_upscale,
            'gaussian': self.methods.gaussian_denoise,
            'sharpen': self.methods.sharpen,
            'unsharp': self.methods.unsharp_mask,
            'combined': self.methods.combined_denoise_sharpen,
        }
        
        # 初始化 Real-ESRGAN（如果可用）
        self.realesrgan = None
        if REALESRGAN_AVAILABLE:
            self.realesrgan = self._init_realesrgan()
        
        # 載入配對信息
        self.pairs_df = self.load_pairs()
    
    def _init_realesrgan(self):
        """初始化 Real-ESRGAN"""
        try:
            if not self.config.MODEL_PATH.exists():
                print(f"⚠️  模型文件不存在: {self.config.MODEL_PATH}")
                return None
            
            print(f"\n📦 載入 Real-ESRGAN 模型...")
            print(f"  設備: {self.config.DEVICE}")
            
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
                model_path=str(self.config.MODEL_PATH),
                model=model,
                tile=self.config.TILE,
                tile_pad=self.config.TILE_PAD,
                pre_pad=self.config.PRE_PAD,
                half=not self.config.FP32,
                device=self.config.DEVICE
            )
            
            print("✓ Real-ESRGAN 模型載入成功\n")
            return upsampler
        
        except Exception as e:
            print(f"⚠️  Real-ESRGAN 初始化失敗: {e}")
            return None
    
    def load_pairs(self) -> pd.DataFrame:
        """載入 pairs.csv"""
        if not self.config.PAIRS_CSV.exists():
            print(f"❌ 配對文件不存在: {self.config.PAIRS_CSV}")
            return pd.DataFrame()
        
        print(f"📁 載入配對信息: {self.config.PAIRS_CSV}")
        df = pd.read_csv(self.config.PAIRS_CSV)
        print(f"✓ 載入 {len(df)} 對配對")
        return df
    
    def merge_results(self):
        """合併所有結果"""
        print("\n" + "=" * 80)
        print("🔗 開始合併基準結果")
        print("=" * 80)
        
        # 載入舊結果（傳統方法）
        old_results_path = self.config.OLD_RESULTS_DIR / "full_results_with_gt.csv"
        if not old_results_path.exists():
            print(f"❌ 找不到舊結果: {old_results_path}")
            return
        
        print(f"\n📂 載入傳統方法結果: {old_results_path}")
        df_old = pd.read_csv(old_results_path)
        print(f"  ✓ 載入 {len(df_old)} 條記錄")
        print(f"  ✓ 包含方法: {', '.join(df_old['method'].unique())}")
        
        # 載入 Real-ESRGAN 結果
        realesrgan_results_path = self.config.REALESRGAN_RESULTS_DIR / "realesrgan_results.csv"
        if not realesrgan_results_path.exists():
            print(f"❌ 找不到 Real-ESRGAN 結果: {realesrgan_results_path}")
            print("請先運行 evaluate_realesrgan_only.py")
            return
        
        print(f"\n📂 載入 Real-ESRGAN 結果: {realesrgan_results_path}")
        df_realesrgan = pd.read_csv(realesrgan_results_path)
        print(f"  ✓ 載入 {len(df_realesrgan)} 條記錄")
        
        # 合併結果
        print(f"\n🔗 合併結果...")
        df_merged = pd.concat([df_old, df_realesrgan], ignore_index=True)
        print(f"  ✓ 合併後共 {len(df_merged)} 條記錄")
        print(f"  ✓ 包含方法: {', '.join(sorted(df_merged['method'].unique()))}")
        
        # 保存合併結果
        merged_path = self.config.LEADERBOARD_DIR / "full_results_merged.csv"
        df_merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 合併結果已保存: {merged_path}")
        
        # 生成排行榜
        self.generate_leaderboards(df_merged)
        
        # 生成統計摘要
        self.generate_statistics_summary(df_merged)
        
        # 識別失敗案例
        self.identify_failure_cases(df_merged)
        
        return df_merged
    
    def generate_leaderboards(self, df: pd.DataFrame):
        """生成排行榜"""
        print("\n📊 生成排行榜...")
        
        # 按退化類型分組
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            # 計算平均指標
            agg_dict = {
                'var_laplacian': 'mean',
                'tenengrad': 'mean',
                'blur_extent': 'mean',
                'luminance': 'mean',
                'contrast': 'mean',
                'entropy': 'mean',
                'gradient_mean': 'mean',
            }
            
            # 添加有參考指標
            if 'psnr' in df_type.columns:
                agg_dict['psnr'] = 'mean'
            if 'ssim' in df_type.columns:
                agg_dict['ssim'] = 'mean'
            if 'lpips' in df_type.columns:
                agg_dict['lpips'] = 'mean'
            
            leaderboard = df_type.groupby('method').agg(agg_dict).round(4)
            
            # 計算綜合得分
            if 'psnr' in leaderboard.columns and 'ssim' in leaderboard.columns:
                # 歸一化 PSNR（假設範圍 20-40 dB）
                psnr_normalized = (leaderboard['psnr'] - 20) / 20
                psnr_normalized = psnr_normalized.clip(0, 1)
                
                # SSIM 已經在 0-1 範圍
                ssim_normalized = leaderboard['ssim']
                
                # 綜合得分（PSNR 50% + SSIM 50%）
                leaderboard['reference_score'] = (psnr_normalized + ssim_normalized) / 2
            
            # 銳度得分
            leaderboard['sharpness_score'] = (
                leaderboard['var_laplacian'] / leaderboard['var_laplacian'].max() +
                leaderboard['tenengrad'] / leaderboard['tenengrad'].max()
            ) / 2
            
            # 模糊得分
            leaderboard['blur_score'] = 1 - (leaderboard['blur_extent'] / leaderboard['blur_extent'].max())
            
            # 總體得分
            if 'reference_score' in leaderboard.columns:
                leaderboard['overall_score'] = (
                    0.6 * leaderboard['reference_score'] +
                    0.2 * leaderboard['sharpness_score'] +
                    0.2 * leaderboard['blur_score']
                )
            else:
                leaderboard['overall_score'] = (
                    leaderboard['sharpness_score'] + leaderboard['blur_score']
                ) / 2
            
            # 排序
            leaderboard = leaderboard.sort_values('overall_score', ascending=False)
            
            # 保存
            deg_type_safe = deg_type.replace(' ', '_').replace('-', '_').lower()
            output_path = self.config.LEADERBOARD_DIR / f"leaderboard_{deg_type_safe}.csv"
            leaderboard.to_csv(output_path, encoding='utf-8-sig')
            print(f"  ✓ {deg_type} 排行榜已保存: {output_path.name}")
            
            # 打印前5名（包含 Real-ESRGAN）
            print(f"\n  🏆 {deg_type} Top 5:")
            for i, (method, row) in enumerate(leaderboard.head(5).iterrows(), 1):
                method_display = self.config.METHOD_DISPLAY_NAMES.get(method, method)
                psnr_str = f"PSNR: {row['psnr']:.2f} dB, " if 'psnr' in row else ""
                ssim_str = f"SSIM: {row['ssim']:.4f}, " if 'ssim' in row else ""
                print(f"    {i}. {method_display:15s} ({psnr_str}{ssim_str}Overall: {row['overall_score']:.4f})")
    
    def generate_statistics_summary(self, df: pd.DataFrame):
        """生成統計摘要"""
        print("\n📈 生成統計摘要...")
        
        summary = {
            'total_images': len(df['image_name'].unique()),
            'total_evaluations': len(df),
            'methods': list(df['method'].unique()),
            'degradation_types': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 按退化類型統計
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            # 找最佳方法
            if 'psnr' in df_type.columns:
                best_method_psnr = df_type.groupby('method')['psnr'].mean().idxmax()
                best_psnr = df_type.groupby('method')['psnr'].mean().max()
            else:
                best_method_psnr = 'N/A'
                best_psnr = 0
            
            summary['degradation_types'][deg_type] = {
                'count': len(df_type['image_name'].unique()),
                'best_method': best_method_psnr,
                'best_psnr': float(best_psnr) if best_psnr > 0 else None
            }
        
        # 保存摘要
        summary_path = self.config.LEADERBOARD_DIR / "evaluation_summary_merged.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 統計摘要已保存: {summary_path.name}")
    
    def identify_failure_cases(self, df: pd.DataFrame):
        """識別失敗案例"""
        print(f"\n🔍 識別失敗案例...")
        
        if 'psnr' not in df.columns:
            print("  ⚠️  沒有 PSNR 數據，跳過失敗案例分析")
            return
        
        # 對於每個方法，找出 PSNR 最低的案例
        all_methods = df['method'].unique()
        for method_name in all_methods:
            df_method = df[df['method'] == method_name]
            
            if df_method.empty:
                continue
            
            # 找出最差的案例
            worst_cases = df_method.nsmallest(self.config.NUM_FAILURE_SAMPLES, 'psnr')
            
            # 保存
            failure_list_path = self.config.FAILURE_DIR / f"failure_cases_{method_name}.csv"
            worst_cases.to_csv(failure_list_path, index=False, encoding='utf-8-sig')
        
        print(f"  ✓ 失敗案例已保存至: {self.config.FAILURE_DIR}")
    
    def generate_comparison_samples(self, num_samples: int = 50):
        """生成對比樣本（50 張，包含所有 9 個方法）"""
        print(f"\n🖼️  生成對比樣本（{num_samples} 張，包含所有方法）...")
        
        if self.pairs_df.empty:
            print("❌ 沒有配對信息")
            return
        
        # 選擇前 num_samples 張圖像
        sample_pairs = self.pairs_df.head(num_samples)
        
        for idx, row in tqdm(sample_pairs.iterrows(), total=len(sample_pairs), desc="生成對比圖"):
            degraded_path = Path(row['degraded_path'])
            original_path = Path(row['target_path'])
            
            if not degraded_path.exists() or not original_path.exists():
                continue
            
            # 讀取圖像
            degraded = cv2.imread(str(degraded_path))
            original = cv2.imread(str(original_path))
            
            if degraded is None or original is None:
                continue
            
            # 確保尺寸一致
            if degraded.shape != original.shape:
                degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
            
            # 應用所有方法
            results = {
                'Original (GT)': original,
                'Degraded': degraded
            }
            
            # 傳統方法
            for method_name, method_func in self.baseline_methods.items():
                try:
                    enhanced = method_func(degraded)
                    results[method_name] = enhanced
                except Exception as e:
                    print(f"  ⚠️  {method_name} 失敗: {e}")
                    continue
            
            # Real-ESRGAN
            if self.realesrgan is not None:
                try:
                    enhanced, _ = self.realesrgan.enhance(degraded, outscale=self.config.OUTSCALE)
                    if enhanced.shape[:2] != degraded.shape[:2]:
                        enhanced = cv2.resize(enhanced, (degraded.shape[1], degraded.shape[0]))
                    results['realesrgan'] = enhanced
                except Exception as e:
                    print(f"  ⚠️  Real-ESRGAN 失敗: {e}")
            
            # 創建對比圖（3 行 3 列 = 9 張圖）
            num_methods = len(results)
            cols = 3
            rows = (num_methods + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten()
            
            # 按順序排列
            plot_idx = 0
            for method_key in self.config.METHOD_ORDER:
                if method_key in results and plot_idx < len(axes):
                    img = results[method_key]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    axes[plot_idx].imshow(img_rgb)
                    
                    # 設置標題
                    if method_key in ['Original (GT)', 'Degraded']:
                        title = method_key
                    else:
                        title = self.config.METHOD_DISPLAY_NAMES.get(method_key, method_key)
                    
                    axes[plot_idx].set_title(title, fontsize=12, fontweight='bold')
                    axes[plot_idx].axis('off')
                    plot_idx += 1
            
            # 隱藏多餘的子圖
            for idx_img in range(plot_idx, len(axes)):
                axes[idx_img].axis('off')
            
            plt.tight_layout()
            
            # 保存
            output_name = f"{degraded_path.stem}_comparison_full.png"
            output_path = self.config.COMPARISON_DIR / output_name
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 對比樣本已保存至: {self.config.COMPARISON_DIR}")


# ==================== 主程序 ====================

def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("🚀 合併基準結果並生成完整對比圖")
    print("=" * 80)
    print("\n📋 配置資訊:")
    print(f"  舊結果目錄: {MergeConfig.OLD_RESULTS_DIR}")
    print(f"  Real-ESRGAN 結果目錄: {MergeConfig.REALESRGAN_RESULTS_DIR}")
    print(f"  輸出目錄: {MergeConfig.OUTPUT_DIR}")
    print(f"  對比樣本數量: {MergeConfig.NUM_SAMPLES}")
    print(f"  Real-ESRGAN 可用: {'是' if REALESRGAN_AVAILABLE else '否'}")
    
    # 創建合併器
    merger = BaselineResultsMerger()
    
    # 合併結果
    df_merged = merger.merge_results()
    
    if df_merged is not None:
        # 生成對比樣本
        merger.generate_comparison_samples(MergeConfig.NUM_SAMPLES)
        
        print("\n" + "=" * 80)
        print("✅ 所有任務完成！")
        print("=" * 80)
        print(f"\n📁 所有結果已保存至: {MergeConfig.OUTPUT_DIR}")
        print(f"\n📊 包含方法: {len(df_merged['method'].unique())} 個")
        print(f"   {', '.join(sorted(df_merged['method'].unique()))}")
        print(f"\n🖼️  對比圖數量: {MergeConfig.NUM_SAMPLES} 張")
        print(f"   每張包含 9 個子圖（GT + Degraded + 7 個方法）")


if __name__ == "__main__":
    main()

