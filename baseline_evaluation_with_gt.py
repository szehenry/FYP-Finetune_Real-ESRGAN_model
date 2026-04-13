#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基準測試與完整性檢查 - 有參考評估版本
========================================

改進點：
1. ✅ 使用 pairs.csv 建立退化-原始圖像配對
2. ✅ 計算有參考指標（PSNR, SSIM, LPIPS）
3. ✅ 同時計算無參考指標
4. ✅ 支持實際的目錄結構（degraded/）
5. ✅ 正確處理退化類型（global_blur, object_blur, low_light）

作者：FYP Project
日期：2025-11
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm
import warnings
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


# ==================== 主評估器（有參考版本） ====================

class BaselineEvaluatorWithGT:
    """基準評估器 - 支援有參考評估"""
    
    def __init__(self):
        self.config = ConfigWithGT()
        self.methods = BaselineMethods()
        self.calculator = MetricsCalculator()
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 基準方法字典
        self.baseline_methods = {
            'identity': self.methods.identity,
            'bicubic': self.methods.bicubic_upscale,
            'gaussian': self.methods.gaussian_denoise,
            # 'bilateral': self.methods.bilateral_denoise,
            # 'nlm': self.methods.nlm_denoise,
            'sharpen': self.methods.sharpen,
            'unsharp': self.methods.unsharp_mask,
            'combined': self.methods.combined_denoise_sharpen,
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
        print("🎯 開始基準測試評估（有參考指標）")
        print("=" * 80)
        
        if self.pairs_df.empty:
            print("❌ 沒有配對信息，無法進行有參考評估")
            return
        
        # 初始化結果存儲
        all_results = []
        
        # 處理每對圖像
        print(f"\n📊 處理 {len(self.pairs_df)} 對圖像...")
        for idx, row in tqdm(self.pairs_df.iterrows(), total=len(self.pairs_df), desc="評估進度"):
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
            
            # 對每個基準方法
            for method_name, method_func in self.baseline_methods.items():
                try:
                    # 應用基準方法
                    enhanced = method_func(degraded)
                    
                    # 🔴 關鍵：計算有參考指標（與原始圖像比較）
                    metrics = self.calculator.evaluate_with_reference(enhanced, original)
                    
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
            
            # 添加無參考指標（如果有）
            if 'niqe' in df_type.columns:
                agg_dict['niqe'] = 'mean'
            if 'brisque' in df_type.columns:
                agg_dict['brisque'] = 'mean'
            
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
    
    def evaluate_single_image(self, input_path: Path, gt_path: Optional[Path], output_dir: Path):
        """對單張圖像執行所有基準方法並輸出對比圖"""
        import matplotlib.pyplot as plt

        print(f"\n📷 輸入圖像: {input_path}")
        if gt_path:
            print(f"🎯 Ground Truth: {gt_path}")
        print(f"📁 輸出目錄: {output_dir}")

        # 讀取退化圖像
        degraded = cv2.imread(str(input_path))
        if degraded is None:
            print(f"❌ 無法讀取圖像: {input_path}")
            return

        # 讀取 GT（如果提供）
        original = None
        if gt_path is not None:
            original = cv2.imread(str(gt_path))
            if original is None:
                print(f"⚠️  無法讀取 GT 圖像: {gt_path}，將跳過有參考指標")
            elif degraded.shape != original.shape:
                print(f"  ⚠️  尺寸不匹配，自動 resize 退化圖像")
                degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))

        # 應用所有基準方法
        panels = {}
        if original is not None:
            panels['Original (GT)'] = original
        panels['Degraded (Input)'] = degraded

        metrics_rows = []
        for method_name, method_func in self.baseline_methods.items():
            try:
                enhanced = method_func(degraded)
                panels[method_name] = enhanced

                if original is not None:
                    m = self.calculator.evaluate_with_reference(enhanced, original)
                    metrics_rows.append({'method': method_name, **m})
                    psnr_str = f"  PSNR={m.get('psnr', 0):.2f} dB  SSIM={m.get('ssim', 0):.4f}"
                    print(f"  [{method_name:10s}]{psnr_str}")
            except Exception as e:
                print(f"  ⚠️  {method_name} 失敗: {e}")

        # 建立對比圖
        num_panels = len(panels)
        cols = 4
        rows = (num_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if num_panels > 1 else [axes]

        for i, (name, img) in enumerate(panels.items()):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)

            # 在標題加上指標
            title = name
            if original is not None and name not in ('Original (GT)', 'Degraded (Input)'):
                row = next((r for r in metrics_rows if r['method'] == name), None)
                if row:
                    title += f"\nPSNR={row.get('psnr', 0):.2f} SSIM={row.get('ssim', 0):.4f}"
            axes[i].set_title(title, fontsize=9)
            axes[i].axis('off')

        for i in range(num_panels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Baseline Comparison — {input_path.name}", fontsize=12, fontweight='bold')
        plt.tight_layout()

        # 儲存
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_path.stem}_comparison.png"
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\n✅ 對比圖已儲存: {output_file}")

        # 若有指標，也儲存 CSV
        if metrics_rows:
            metrics_csv = output_dir / f"{input_path.stem}_metrics.csv"
            pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 指標 CSV 已儲存: {metrics_csv}")


# ==================== 主程序 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="基準測試評估（有參考指標）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "範例用法：\n"
            "  # 單張圖像模式（無 GT）\n"
            "  python baseline_evaluation_with_gt.py --input /path/to/degraded.jpg\n\n"
            "  # 單張圖像模式（有 GT）\n"
            "  python baseline_evaluation_with_gt.py --input /path/to/degraded.jpg --gt /path/to/original.jpg\n\n"
            "  # 自訂輸出目錄\n"
            "  python baseline_evaluation_with_gt.py --input /path/to/degraded.jpg \\\n"
            "      --output-dir '/Volumes/Extreme SSD/baseline_results_old/comparison_samples'\n\n"
            "  # 批次模式（使用 pairs.csv）\n"
            "  python baseline_evaluation_with_gt.py --batch"
        )
    )
    parser.add_argument(
        '--input', '-i', type=str, default=None,
        help='輸入退化圖像路徑（單張圖像模式）'
    )
    parser.add_argument(
        '--gt', '-g', type=str, default=None,
        help='Ground Truth 原始圖像路徑（可選，用於計算 PSNR/SSIM）'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str,
        default='/Volumes/Extreme SSD/baseline_results_old/comparison_samples',
        help='對比圖輸出目錄（預設：/Volumes/Extreme SSD/baseline_results_old/comparison_samples）'
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='批次模式：使用 pairs.csv 評估所有圖像（預設行為）'
    )
    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    print("\n" + "=" * 80)
    print("🚀 基準測試與完整性檢查（有參考評估）")
    print("=" * 80)

    # ── 單張圖像模式 ──────────────────────────────────────────────
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 找不到輸入圖像: {input_path}")
            sys.exit(1)

        gt_path = Path(args.gt) if args.gt else None
        output_dir = Path(args.output_dir)

        evaluator = BaselineEvaluatorWithGT()
        evaluator.evaluate_single_image(input_path, gt_path, output_dir)
        return

    # ── 批次模式 ──────────────────────────────────────────────────
    print("\n📋 配置資訊:")
    print(f"  退化圖像目錄: {ConfigWithGT.DEGRADED_DIR}")
    print(f"  原始圖像目錄: {ConfigWithGT.ORIGINAL_DIR}")
    print(f"  配對文件: {ConfigWithGT.PAIRS_CSV}")
    print(f"  輸出目錄: {ConfigWithGT.OUTPUT_DIR}")
    print(f"  設備: {ConfigWithGT.DEVICE}")
    print(f"  LPIPS 可用: {'是' if LPIPS_AVAILABLE else '否'}")
    print(f"  PYIQA 可用: {'是' if PYIQA_AVAILABLE else '否'}")

    evaluator = BaselineEvaluatorWithGT()
    evaluator.evaluate_all_baselines()

    print("\n✅ 所有任務完成！")
    print(f"\n📁 請檢查輸出目錄: {ConfigWithGT.OUTPUT_DIR}")
    print("\n💡 提示：排行榜中包含 PSNR/SSIM/LPIPS 等有參考指標")


if __name__ == "__main__":
    main()

