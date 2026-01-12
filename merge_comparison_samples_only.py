#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合併對比樣本生成器
==================

功能：
1. 保持各方法的 Leaderboards 獨立
2. 只合併生成新的 Comparison Samples（9 個方法）
3. Enhanced images 只在 Real-ESRGAN 資料夾

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
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

# 設置 UTF-8 輸出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 導入基準方法
try:
    from baseline_evaluation import BaselineMethods
    print("✓ 成功導入基準方法模組")
except ImportError as e:
    print(f"❌ 無法導入 baseline_evaluation 模組: {e}")
    sys.exit(1)


# ==================== 配置區 ====================

class MergeConfig:
    """合併配置"""
    
    # 輸入路徑
    BASELINE_RESULTS_DIR = Path(r"D:\baseline_results_old\leaderboards")
    REALESRGAN_RESULTS_DIR = Path(r"D:\baseline_results_realesrgan_only")
    
    BASELINE_CSV = BASELINE_RESULTS_DIR / "full_results_with_gt.csv"
    REALESRGAN_CSV = REALESRGAN_RESULTS_DIR / "leaderboards" / "realesrgan_results.csv"
    REALESRGAN_ENHANCED_DIR = REALESRGAN_RESULTS_DIR / "enhanced_images"
    
    # 輸出路徑（只生成對比樣本）
    OUTPUT_DIR = Path(r"D:\baseline_results_merged")
    COMPARISON_DIR = OUTPUT_DIR / "comparison_samples"
    
    # 對比樣本配置
    NUM_SAMPLES = 50  # 生成 50 個對比樣本
    
    # 圖像路徑
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset\degraded")
    ORIGINAL_DIR = Path(r"D:\FYP_Images")
    
    # 退化類型映射
    DEGRADATION_TYPE_MAP = {
        'global_blur': 'Drone Motion Blur',
        'object_blur': 'Object Motion Blur',
        'low_light': 'Low-light'
    }


# ==================== 對比樣本生成器 ====================

class ComparisonSampleGenerator:
    """生成包含所有方法的對比樣本"""
    
    def __init__(self):
        self.config = MergeConfig()
        self.methods = BaselineMethods()
        
        # 基準方法字典（與 HenryCC 一致）
        self.baseline_methods = {
            'identity': self.methods.identity,
            'bicubic': self.methods.bicubic_upscale,
            'gaussian': self.methods.gaussian_denoise,
            'sharpen': self.methods.sharpen,
            'unsharp': self.methods.unsharp_mask,
            'combined': self.methods.combined_denoise_sharpen,
        }
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_results(self):
        """載入 Baseline 和 Real-ESRGAN 結果"""
        print("\n📁 載入結果文件...")
        
        # 載入 Baseline 結果
        if not self.config.BASELINE_CSV.exists():
            print(f"❌ Baseline 結果不存在: {self.config.BASELINE_CSV}")
            return None, None
        
        df_baseline = pd.read_csv(self.config.BASELINE_CSV)
        print(f"✓ Baseline 結果: {len(df_baseline)} 條記錄")
        print(f"  方法: {df_baseline['method'].unique().tolist()}")
        
        # 載入 Real-ESRGAN 結果
        if not self.config.REALESRGAN_CSV.exists():
            print(f"❌ Real-ESRGAN 結果不存在: {self.config.REALESRGAN_CSV}")
            return df_baseline, None
        
        df_realesrgan = pd.read_csv(self.config.REALESRGAN_CSV)
        print(f"✓ Real-ESRGAN 結果: {len(df_realesrgan)} 條記錄")
        
        return df_baseline, df_realesrgan
    
    def get_sample_images(self, df_baseline: pd.DataFrame) -> List[Dict]:
        """選擇要生成對比樣本的圖像"""
        print(f"\n🎲 選擇 {self.config.NUM_SAMPLES} 張圖像生成對比樣本...")
        
        # 從 Baseline 結果中隨機選擇（已經評估過的圖像）
        unique_images = df_baseline[['image_name', 'degradation_type']].drop_duplicates()
        
        # 分層採樣：每種退化類型均勻採樣
        samples = []
        samples_per_type = self.config.NUM_SAMPLES // len(unique_images['degradation_type'].unique())
        
        for deg_type in unique_images['degradation_type'].unique():
            type_images = unique_images[unique_images['degradation_type'] == deg_type]
            sampled = type_images.sample(n=min(samples_per_type, len(type_images)), random_state=42)
            samples.extend(sampled.to_dict('records'))
        
        # 如果不夠，隨機補充
        if len(samples) < self.config.NUM_SAMPLES:
            remaining = self.config.NUM_SAMPLES - len(samples)
            remaining_images = unique_images[~unique_images['image_name'].isin([s['image_name'] for s in samples])]
            if len(remaining_images) > 0:
                extra = remaining_images.sample(n=min(remaining, len(remaining_images)), random_state=42)
                samples.extend(extra.to_dict('records'))
        
        print(f"✓ 已選擇 {len(samples)} 張圖像")
        for deg_type in unique_images['degradation_type'].unique():
            count = sum(1 for s in samples if s['degradation_type'] == deg_type)
            print(f"  {deg_type}: {count} 張")
        
        return samples[:self.config.NUM_SAMPLES]
    
    def generate_comparison_samples(self, sample_list: List[Dict]):
        """生成對比樣本（9 個方法）"""
        print(f"\n🖼️  生成對比樣本...")
        
        for idx, sample in enumerate(tqdm(sample_list, desc="生成樣本")):
            image_name = sample['image_name']
            deg_type = sample['degradation_type']
            
            # 構建路徑
            degraded_path = self.config.DEGRADED_DIR / image_name
            
            # 找到對應的原始圖像
            original_name = self.find_original_image(image_name)
            if original_name is None:
                print(f"\n  ⚠️  找不到原始圖像: {image_name}")
                continue
            
            original_path = self.config.ORIGINAL_DIR / original_name
            
            # 檢查文件存在
            if not degraded_path.exists():
                print(f"\n  ⚠️  退化圖像不存在: {degraded_path}")
                continue
            
            if not original_path.exists():
                print(f"\n  ⚠️  原始圖像不存在: {original_path}")
                continue
            
            # 讀取圖像
            degraded = cv2.imread(str(degraded_path))
            original = cv2.imread(str(original_path))
            
            if degraded is None or original is None:
                continue
            
            # 確保尺寸一致
            if degraded.shape != original.shape:
                degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
            
            # 準備結果字典
            results = {
                'Original (GT)': original,
                'Degraded': degraded
            }
            
            # 應用傳統基準方法
            for method_name, method_func in self.baseline_methods.items():
                try:
                    enhanced = method_func(degraded)
                    results[method_name] = enhanced
                except Exception as e:
                    print(f"\n  ⚠️  {method_name} 處理失敗: {e}")
                    continue
            
            # 添加 Real-ESRGAN 結果（從保存的增強圖像讀取）
            realesrgan_enhanced_path = self.config.REALESRGAN_ENHANCED_DIR / f"enhanced_{Path(image_name).stem}.png"
            if realesrgan_enhanced_path.exists():
                realesrgan_enhanced = cv2.imread(str(realesrgan_enhanced_path))
                if realesrgan_enhanced is not None:
                    # 確保尺寸一致
                    if realesrgan_enhanced.shape != original.shape:
                        realesrgan_enhanced = cv2.resize(realesrgan_enhanced, (original.shape[1], original.shape[0]))
                    results['realesrgan'] = realesrgan_enhanced
                else:
                    print(f"\n  ⚠️  無法讀取 Real-ESRGAN 增強圖像: {realesrgan_enhanced_path}")
            else:
                print(f"\n  ⚠️  Real-ESRGAN 增強圖像不存在: {realesrgan_enhanced_path}")
                # 如果沒有保存的增強圖像，跳過這個樣本
                continue
            
            # 創建對比圖（3 rows × 3 cols = 9 個方法）
            num_methods = len(results)
            cols = 3
            rows = (num_methods + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten()
            
            for idx_img, (name, img) in enumerate(results.items()):
                if idx_img >= len(axes):
                    break
                
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[idx_img].imshow(img_rgb)
                axes[idx_img].set_title(name, fontsize=12, fontweight='bold')
                axes[idx_img].axis('off')
            
            # 隱藏多餘的子圖
            for idx_img in range(num_methods, len(axes)):
                axes[idx_img].axis('off')
            
            # 添加整體標題
            fig.suptitle(f"{deg_type} - {image_name}", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存
            output_name = f"{Path(image_name).stem}_comparison.png"
            output_path = self.config.COMPARISON_DIR / output_name
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ 對比樣本已保存至: {self.config.COMPARISON_DIR}")
    
    def find_original_image(self, degraded_name: str) -> str:
        """從退化圖像名找到原始圖像名"""
        # 移除退化後綴
        # 例如：0000001_00012_d_0000001_global_blur.jpg -> 0000001.jpg
        
        parts = degraded_name.split('_d_')
        if len(parts) >= 2:
            original_id = parts[0].split('_')[0]  # 取第一個部分的第一段
            
            # 在原始目錄中搜索
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                potential_path = self.config.ORIGINAL_DIR / f"{original_id}{ext}"
                if potential_path.exists():
                    return potential_path.name
        
        return None
    
    def run(self):
        """執行合併流程"""
        print("\n" + "=" * 80)
        print("🎯 生成合併對比樣本（保持 Leaderboards 獨立）")
        print("=" * 80)
        
        # 載入結果
        df_baseline, df_realesrgan = self.load_results()
        
        if df_baseline is None:
            print("\n❌ 缺少必要的結果文件")
            return
        
        if df_realesrgan is None:
            print("\n⚠️  沒有 Real-ESRGAN 結果，只使用 Baseline")
        
        # 選擇樣本圖像
        sample_list = self.get_sample_images(df_baseline)
        
        # 生成對比樣本
        self.generate_comparison_samples(sample_list)
        
        print("\n" + "=" * 80)
        print("✅ 合併完成！")
        print("=" * 80)
        print(f"\n📁 輸出位置:")
        print(f"  對比樣本: {self.config.COMPARISON_DIR}")
        print(f"\n💡 說明:")
        print(f"  ✅ Leaderboards 保持獨立（未合併）")
        print(f"  ✅ 對比樣本包含 9 個方法")
        print(f"  ✅ Enhanced images 只在 Real-ESRGAN 資料夾")


# ==================== 主程序 ====================

def main():
    """主函數"""
    generator = ComparisonSampleGenerator()
    generator.run()


if __name__ == "__main__":
    main()

