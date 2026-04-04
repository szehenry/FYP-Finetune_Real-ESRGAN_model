#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基準測試與完整性檢查 - 有參考評估版本
========================================

改進點：
1. 使用 pairs.csv 建立退化-原始圖像配對
2. 計算有參考指標（PSNR, SSIM, LPIPS）
3. 同時計算無參考指標
4. 支持實際的目錄結構（degraded/）
5. 正確處理退化類型（global_blur, object_blur, low_light）

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
import json
from datetime import datetime
from tqdm import tqdm
import warnings

# 設置 UTF-8 輸出（修復 Windows 編碼問題）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 嘗試導入進階指標庫
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False

# 設置隨機種子（確保可重現）
import random
random.seed(42)
np.random.seed(42)

# 導入基準方法（從原始文件）
try:
    from baseline_evaluation import (
        BaselineMethods, MetricsCalculator, Config as BaseConfig
    )
    print("✓ 成功導入基準方法模組")
except ImportError as e:
    print(f"❌ 無法導入 baseline_evaluation 模組: {e}")
    print("請確保 baseline_evaluation.py 在相同目錄下")
    import sys
    sys.exit(1)


# ==================== 配置區 ====================

class ConfigWithGT(BaseConfig):
    """擴展配置類，添加 Ground Truth 支持"""
    
    # 額外的路徑
    PAIRS_CSV = Path(r"D:\degraded_full_dataset\pairs.csv")
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset\degraded")  # 更正：扁平結構
    ORIGINAL_DIR = Path(r"D:\FYP_Images")
    
    # 退化類型映射（代碼名 -> 顯示名）
    DEGRADATION_TYPE_MAP = {
        'global_blur': 'Drone Motion Blur',
        'object_blur': 'Object Motion Blur',
        'low_light': 'Low-light'
    }
    
    # LPIPS 抽樣配置
    LPIPS_SAMPLE_SIZE = 700  # 隨機抽樣 700 張計算 LPIPS
    LPIPS_ENABLED = True  # 是否啟用 LPIPS


# ==================== 主評估器（有參考版本） ====================

class BaselineEvaluatorWithGT:
    """基準評估器 - 支援有參考評估"""
    
    def __init__(self):
        self.config = ConfigWithGT()
        self.methods = BaselineMethods()
        self.calculator = MetricsCalculator(enable_lpips=self.config.LPIPS_ENABLED)
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 基準方法字典（優化後配置）
        self.baseline_methods = {
            'identity': self.methods.identity,          # 必須：基準線
            'bicubic': self.methods.bicubic_upscale,   # 必須：學術標準
            'gaussian': self.methods.gaussian_denoise, # 推薦：快速去噪
            # 'bilateral': self.methods.bilateral_denoise,  # 移除：較慢
            # 'nlm': self.methods.nlm_denoise,  # 移除：非常慢且不常用
            'sharpen': self.methods.sharpen,           # 推薦：傳統銳化
            'unsharp': self.methods.unsharp_mask,      # 推薦：更好的銳化
            'combined': self.methods.combined_denoise_sharpen,  # 必須：最佳傳統
        }
        
        # 載入配對信息
        self.pairs_df = self.load_pairs()
    
    def load_pairs(self) -> pd.DataFrame:
        """載入 pairs.csv"""
        if not self.config.PAIRS_CSV.exists():
            print(f"❌ 配對文件不存在: {self.config.PAIRS_CSV}")
            return pd.DataFrame()
        
        print(f"📁 載入配對信息: {self.config.PAIRS_CSV}")
        df = pd.read_csv(self.config.PAIRS_CSV)
        print(f"✓ 載入 {len(df)} 對配對")
        
        # 顯示退化類型統計
        print("\n退化類型統計:")
        for mode in df['mode'].unique():
            count = len(df[df['mode'] == mode])
            display_name = self.config.DEGRADATION_TYPE_MAP.get(mode, mode)
            print(f"  {display_name}: {count} 張")
        
        return df
    
    def evaluate_all_baselines(self):
        """評估所有基準方法 - 有參考版本"""
        print("\n" + "=" * 80)
        print("🎯 開始基準測試評估（有參考指標 + LPIPS 隨機抽樣）")
        print("=" * 80)
        
        if self.pairs_df.empty:
            print("❌ 沒有配對信息，無法進行有參考評估")
            return
        
        # 隨機選擇要計算 LPIPS 的圖像索引
        total_images = len(self.pairs_df)
        sample_size = min(self.config.LPIPS_SAMPLE_SIZE, total_images)
        lpips_sample_indices = set(random.sample(range(total_images), sample_size))
        
        print(f"\n💡 優化策略:")
        print(f"  ✅ PSNR/SSIM: 所有 {total_images} 張圖像")
        print(f"  ✅ LPIPS: 隨機抽樣 {sample_size} 張 ({sample_size/total_images*100:.1f}%)")
        print(f"  ⏭️  NIQE/BRISQUE: 已跳過（我們有 Ground Truth）")
        print(f"  ⏭️  NLM/Bilateral: 已移除（太慢且不常用）")
        print(f"  🔧 記憶體管理: LPIPS 自動清理 GPU 快取")
        print(f"\n📈 預計時間: ~4-6 小時（6個方法 vs 原本8個）")
        
        # 初始化結果存儲
        all_results = []
        
        # 獲取最後一個方法名（用於記憶體清理判斷）
        last_method = list(self.baseline_methods.keys())[-1]
        
        # 處理每對圖像
        print(f"\n📊 處理 {len(self.pairs_df)} 對圖像...")
        for enum_idx, (idx, row) in enumerate(tqdm(self.pairs_df.iterrows(), total=len(self.pairs_df), desc="評估進度")):
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
                print(f"  ⚠️  尺寸不匹配: {degraded_path.name}")
                degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
            
            # 判斷是否對這張圖計算 LPIPS（使用 enum_idx 確保從0開始的連續索引）
            calculate_lpips_for_this_image = enum_idx in lpips_sample_indices
            
            # 對每個基準方法
            for method_name, method_func in self.baseline_methods.items():
                try:
                    # 應用基準方法
                    enhanced = method_func(degraded)
                    
                    # 🔴 關鍵：計算有參考指標（與原始圖像比較）
                    # 僅對抽樣的圖像計算 LPIPS
                    metrics = self.calculator.evaluate_with_reference(
                        enhanced, original, 
                        calculate_lpips=calculate_lpips_for_this_image
                    )
                    
                    # 記錄結果
                    result_entry = {
                        'image_name': degraded_path.name,
                        'degradation_type': self.config.DEGRADATION_TYPE_MAP.get(deg_type, deg_type),
                        'degradation_mode': deg_type,
                        'method': method_name,
                        'split': row['split'],
                        **metrics
                    }
                    all_results.append(result_entry)
                    
                    # 🔧 定期清理記憶體（每處理完一張圖的所有方法後）
                    if method_name == last_method:
                        # 清理 OpenCV 可能的記憶體累積
                        del degraded, original, enhanced
                        # 只有在 torch 可用時才清理 CUDA 快取
                        if LPIPS_AVAILABLE:
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except:
                                pass
                
                except Exception as e:
                    print(f"  ⚠️  {method_name} 處理 {degraded_path.name} 失敗: {e}")
                    continue
        
        if not all_results:
            print("❌ 沒有生成任何結果")
            return
        
        # 保存結果到 DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存完整結果
        full_results_path = self.config.LEADERBOARD_DIR / "full_results_with_gt.csv"
        df.to_csv(full_results_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 完整結果已保存: {full_results_path}")
        
        # 生成排行榜
        self.generate_leaderboards(df)
        
        # 生成統計摘要
        self.generate_statistics_summary(df)
        
        # 生成可視化樣本
        sample_pairs = self.pairs_df.head(self.config.NUM_SAMPLES)
        self.generate_comparison_samples(sample_pairs)
        
        # 識別失敗案例
        self.identify_failure_cases(df)
        
        print("\n" + "=" * 80)
        print("✅ 基準測試評估完成！")
        print("=" * 80)
        print(f"\n📁 所有結果已保存至: {self.config.OUTPUT_DIR}")
    
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
            
            # NIQE/BRISQUE 已移除
            
            leaderboard = df_type.groupby('method').agg(agg_dict).round(4)
            
            # 計算綜合得分
            # 1. PSNR/SSIM 得分（如果有）
            if 'psnr' in leaderboard.columns and 'ssim' in leaderboard.columns:
                # 歸一化 PSNR（假設範圍 20-40 dB）
                psnr_normalized = (leaderboard['psnr'] - 20) / 20
                psnr_normalized = psnr_normalized.clip(0, 1)
                
                # SSIM 已經在 0-1 範圍
                ssim_normalized = leaderboard['ssim']
                
                # 綜合得分（PSNR 50% + SSIM 50%）
                leaderboard['reference_score'] = (psnr_normalized + ssim_normalized) / 2
            
            # 2. 銳度得分
            leaderboard['sharpness_score'] = (
                leaderboard['var_laplacian'] / leaderboard['var_laplacian'].max() +
                leaderboard['tenengrad'] / leaderboard['tenengrad'].max()
            ) / 2
            
            # 3. 模糊得分
            leaderboard['blur_score'] = 1 - (leaderboard['blur_extent'] / leaderboard['blur_extent'].max())
            
            # 4. 總體得分
            if 'reference_score' in leaderboard.columns:
                # 如果有參考指標，參考得分佔 60%，其他佔 40%
                leaderboard['overall_score'] = (
                    0.6 * leaderboard['reference_score'] +
                    0.2 * leaderboard['sharpness_score'] +
                    0.2 * leaderboard['blur_score']
                )
            else:
                # 只有無參考指標
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
            
            # 打印前3名
            print(f"\n  🏆 {deg_type} Top 3:")
            for i, (method, row) in enumerate(leaderboard.head(3).iterrows(), 1):
                psnr_str = f"PSNR: {row['psnr']:.2f} dB, " if 'psnr' in row else ""
                ssim_str = f"SSIM: {row['ssim']:.4f}, " if 'ssim' in row else ""
                print(f"    {i}. {method:12s} ({psnr_str}{ssim_str}Overall: {row['overall_score']:.4f})")
    
    def generate_statistics_summary(self, df: pd.DataFrame):
        """生成統計摘要"""
        print("\n📈 生成統計摘要...")
        
        summary = {
            'total_images': len(df['image_name'].unique()),
            'total_evaluations': len(df),
            'degradation_types': {},
            'methods': list(df['method'].unique()),
            'metrics': {}
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
        summary_path = self.config.LEADERBOARD_DIR / "evaluation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 統計摘要已保存: {summary_path.name}")
    
    def generate_comparison_samples(self, sample_pairs: pd.DataFrame):
        """生成對比樣本"""
        print(f"\n🖼️  生成對比樣本（前 {len(sample_pairs)} 張）...")
        
        import matplotlib.pyplot as plt
        
        for idx, row in tqdm(sample_pairs.iterrows(), total=len(sample_pairs), desc="生成樣本"):
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
            
            # 應用基準方法
            results = {'Original (GT)': original, 'Degraded': degraded}
            for method_name, method_func in self.baseline_methods.items():
                try:
                    enhanced = method_func(degraded)
                    results[method_name] = enhanced
                except:
                    continue
            
            # 創建對比圖
            num_methods = len(results)
            cols = 4
            rows = (num_methods + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
            axes = axes.flatten() if num_methods > 1 else [axes]
            
            for idx_img, (name, img) in enumerate(results.items()):
                if idx_img >= len(axes):
                    break
                
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[idx_img].imshow(img_rgb)
                axes[idx_img].set_title(name, fontsize=10)
                axes[idx_img].axis('off')
            
            # 隱藏多餘的子圖
            for idx_img in range(num_methods, len(axes)):
                axes[idx_img].axis('off')
            
            plt.tight_layout()
            
            # 保存
            output_name = f"{degraded_path.stem}_comparison.png"
            output_path = self.config.COMPARISON_DIR / output_name
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 對比樣本已保存至: {self.config.COMPARISON_DIR}")
    
    def identify_failure_cases(self, df: pd.DataFrame):
        """識別失敗案例"""
        print(f"\n🔍 識別失敗案例...")
        
        if 'psnr' not in df.columns:
            print("  ⚠️  沒有 PSNR 數據，跳過失敗案例分析")
            return
        
        # 對於每個方法，找出 PSNR 最低的案例
        for method_name in self.baseline_methods.keys():
            df_method = df[df['method'] == method_name]
            
            if df_method.empty:
                continue
            
            # 找出最差的案例
            worst_cases = df_method.nsmallest(self.config.NUM_FAILURE_SAMPLES, 'psnr')
            
            # 保存
            failure_list_path = self.config.FAILURE_DIR / f"failure_cases_{method_name}.csv"
            worst_cases.to_csv(failure_list_path, index=False, encoding='utf-8-sig')
        
        print(f"  ✓ 失敗案例已保存至: {self.config.FAILURE_DIR}")


# ==================== 主程序 ====================

def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("🚀 基準測試與完整性檢查（有參考評估）")
    print("=" * 80)
    print("\n📋 配置資訊:")
    print(f"  退化圖像目錄: {ConfigWithGT.DEGRADED_DIR}")
    print(f"  原始圖像目錄: {ConfigWithGT.ORIGINAL_DIR}")
    print(f"  配對文件: {ConfigWithGT.PAIRS_CSV}")
    print(f"  輸出目錄: {ConfigWithGT.OUTPUT_DIR}")
    print(f"  設備: {ConfigWithGT.DEVICE}")
    print(f"  LPIPS 可用: {'是' if LPIPS_AVAILABLE else '否'} (隨機抽樣 {ConfigWithGT.LPIPS_SAMPLE_SIZE} 張)")
    print(f"  NIQE/BRISQUE: 已停用（我們有 Ground Truth）")
    
    # 創建評估器
    evaluator = BaselineEvaluatorWithGT()
    
    # 執行評估
    evaluator.evaluate_all_baselines()
    
    print("\n✅ 所有任務完成！")
    print(f"\n📁 請檢查輸出目錄: {ConfigWithGT.OUTPUT_DIR}")
    print("\n💡 提示：排行榜中包含 PSNR/SSIM/LPIPS 等有參考指標")


if __name__ == "__main__":
    main()

