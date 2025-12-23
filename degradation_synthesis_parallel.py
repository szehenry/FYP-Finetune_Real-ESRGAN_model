"""
平行化影像降質合成腳本 (Parallel Image Degradation Synthesis Script)
使用多處理器 (multiprocessing) 加速處理

適用於：
- 多核心 CPU 的電腦（如 8 核心以上）
- 處理大量影像時
- 需要更快完成處理

作者: AI Assistant
日期: 2025-11-04
"""

import os
import json
import csv
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count
import functools

# 導入原始的降質合成器
from degradation_synthesis import (
    MotionBlurSynthesizer,
    LowLightSynthesizer,
    YOLO_AVAILABLE
)

# 因為 YOLOv8 在多處理中可能有問題，物體模糊會在單執行緒中處理
if YOLO_AVAILABLE:
    from ultralytics import YOLO


def process_single_image_worker(args):
    """
    工作函數：處理單張影像（用於多處理）
    
    注意：這個函數不包含物體模糊（因為 YOLOv8 在多處理中可能有問題）
    
    Args:
        args: (image_path, output_degraded_dir, output_metadata_dir, split, 
               enable_global_blur, enable_low_light, device)
    
    Returns:
        生成的降質影像資訊列表
    """
    (image_path, output_degraded_dir, output_metadata_dir, split,
     enable_global_blur, enable_low_light, device) = args
    
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    image_name = Path(image_path).stem
    results = []
    
    # 初始化合成器（每個工作處理器都有自己的實例）
    global_blur = MotionBlurSynthesizer(device='cpu')  # 多處理時強制使用 CPU
    low_light = LowLightSynthesizer(device='cpu')
    
    # 1. 全局運動模糊
    if enable_global_blur:
        try:
            degraded, params = global_blur.synthesize(image, {})
            
            degraded_filename = f"{image_name}_global_blur.jpg"
            degraded_path = Path(output_degraded_dir) / degraded_filename
            cv2.imwrite(str(degraded_path), degraded)
            
            metadata_filename = f"{image_name}_global_blur.json"
            metadata_path = Path(output_metadata_dir) / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            results.append({
                'degraded_path': str(degraded_path),
                'target_path': image_path,
                'split': split,
                'mode': 'global_blur',
                'metadata_path': str(metadata_path)
            })
        except Exception as e:
            pass  # 靜默失敗，避免影響其他影像
    
    # 2. 低光照降質
    if enable_low_light:
        try:
            degraded, params = low_light.synthesize(image, {})
            
            degraded_filename = f"{image_name}_low_light.jpg"
            degraded_path = Path(output_degraded_dir) / degraded_filename
            cv2.imwrite(str(degraded_path), degraded)
            
            metadata_filename = f"{image_name}_low_light.json"
            metadata_path = Path(output_metadata_dir) / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            results.append({
                'degraded_path': str(degraded_path),
                'target_path': image_path,
                'split': split,
                'mode': 'low_light',
                'metadata_path': str(metadata_path)
            })
        except Exception as e:
            pass
    
    return results


class ParallelDegradationPipeline:
    """平行化降質處理流程"""
    
    def __init__(self, output_dir: str, num_workers: Optional[int] = None):
        """
        初始化平行化降質流程
        
        Args:
            output_dir: 輸出目錄
            num_workers: 工作處理器數量（None = 自動偵測 CPU 核心數）
        """
        # 設定工作處理器數量
        if num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)  # 保留 1 個核心給系統
        else:
            self.num_workers = num_workers
        
        print(f"\n=== 平行化降質合成流程初始化 ===")
        print(f"CPU 核心數: {cpu_count()}")
        print(f"使用工作處理器數: {self.num_workers}")
        
        # 創建輸出目錄
        self.output_dir = Path(output_dir)
        self.degraded_dir = self.output_dir / 'degraded'
        self.metadata_dir = self.output_dir / 'metadata'
        
        self.degraded_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 用於記錄配對資訊
        self.pairs = []
    
    def process_images_parallel(self, image_paths: List[str], split: str,
                                enable_global_blur: bool = True,
                                enable_low_light: bool = True):
        """
        平行處理多張影像（不包含物體模糊）
        
        Args:
            image_paths: 影像路徑列表
            split: 資料集分割
            enable_global_blur: 啟用全局模糊
            enable_low_light: 啟用低光照
        """
        # 準備參數
        args_list = [
            (img_path, str(self.degraded_dir), str(self.metadata_dir), split,
             enable_global_blur, enable_low_light, 'cpu')
            for img_path in image_paths
        ]
        
        # 使用多處理池
        with Pool(processes=self.num_workers) as pool:
            # 使用 imap 以便顯示進度條
            results_list = list(tqdm(
                pool.imap(process_single_image_worker, args_list),
                total=len(args_list),
                desc=f"平行處理 {split}"
            ))
        
        # 收集結果
        for results in results_list:
            self.pairs.extend(results)
    
    def process_object_blur_sequential(self, image_paths: List[str], split: str):
        """
        循序處理物體模糊（因為 YOLOv8 在多處理中可能有問題）
        
        Args:
            image_paths: 影像路徑列表
            split: 資料集分割
        """
        if not YOLO_AVAILABLE:
            print("YOLOv8 未安裝，跳過物體模糊")
            return
        
        print(f"\n處理物體模糊（循序處理）...")
        
        # 初始化 YOLOv8
        from degradation_synthesis import ObjectMotionBlurSynthesizer
        object_blur = ObjectMotionBlurSynthesizer(device='cpu')
        
        for image_path in tqdm(image_paths, desc=f"物體模糊 {split}"):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                degraded, params = object_blur.synthesize(image, {})
                
                # 只有偵測到物體時才儲存
                if params.get('objects_detected', 0) > 0:
                    image_name = Path(image_path).stem
                    
                    degraded_filename = f"{image_name}_object_blur.jpg"
                    degraded_path = self.degraded_dir / degraded_filename
                    cv2.imwrite(str(degraded_path), degraded)
                    
                    metadata_filename = f"{image_name}_object_blur.json"
                    metadata_path = self.metadata_dir / metadata_filename
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(params, f, indent=2, ensure_ascii=False)
                    
                    self.pairs.append({
                        'degraded_path': str(degraded_path),
                        'target_path': image_path,
                        'split': split,
                        'mode': 'object_blur',
                        'metadata_path': str(metadata_path)
                    })
            except Exception as e:
                pass  # 靜默失敗
    
    def process_dataset(self, split_dir: str,
                       enable_global_blur: bool = True,
                       enable_object_blur: bool = True,
                       enable_low_light: bool = True,
                       max_images: Optional[int] = None):
        """
        處理整個資料集（平行化）
        
        Args:
            split_dir: 資料分割目錄
            enable_global_blur: 啟用全局模糊
            enable_object_blur: 啟用物體模糊
            enable_low_light: 啟用低光照
            max_images: 最大處理影像數
        """
        split_dir = Path(split_dir)
        
        # 讀取各分割的影像列表
        splits = {}
        for split_name in ['train', 'val', 'test']:
            list_file = split_dir / f'{split_name}_list.txt'
            if list_file.exists():
                with open(list_file, 'r', encoding='utf-8') as f:
                    image_paths = [line.strip() for line in f if line.strip()]
                splits[split_name] = image_paths
                print(f"{split_name}: {len(image_paths)} 張影像")
        
        # 處理每個分割
        for split_name, image_paths in splits.items():
            if max_images:
                image_paths = image_paths[:max_images]
            
            print(f"\n=== 處理 {split_name} 集 ===")
            
            # 步驟 1: 平行處理全局模糊和低光照
            if enable_global_blur or enable_low_light:
                self.process_images_parallel(
                    image_paths,
                    split=split_name,
                    enable_global_blur=enable_global_blur,
                    enable_low_light=enable_low_light
                )
            
            # 步驟 2: 循序處理物體模糊
            if enable_object_blur:
                self.process_object_blur_sequential(image_paths, split_name)
        
        # 儲存配對資訊
        self.save_pairs_csv()
        self.save_summary()
    
    def save_pairs_csv(self):
        """儲存配對資訊到 CSV"""
        csv_path = self.output_dir / 'pairs.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['degraded_path', 'target_path', 'split', 'mode', 'metadata_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for pair in self.pairs:
                writer.writerow(pair)
        
        print(f"\n配對資訊已儲存到: {csv_path}")
        print(f"總共生成 {len(self.pairs)} 對降質影像")
    
    def save_summary(self):
        """儲存處理摘要"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_workers': self.num_workers,
            'total_pairs': len(self.pairs),
            'by_split': {},
            'by_mode': {}
        }
        
        # 統計各分割
        for split in ['train', 'val', 'test']:
            count = sum(1 for p in self.pairs if p['split'] == split)
            summary['by_split'][split] = count
        
        # 統計各模式
        for mode in ['global_blur', 'object_blur', 'low_light']:
            count = sum(1 for p in self.pairs if p['mode'] == mode)
            summary['by_mode'][mode] = count
        
        summary_path = self.output_dir / 'degradation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"摘要資訊已儲存到: {summary_path}")
        print("\n=== 處理摘要 ===")
        print(f"總降質影像對數: {summary['total_pairs']}")
        print(f"按分割統計: {summary['by_split']}")
        print(f"按模式統計: {summary['by_mode']}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='平行化影像降質合成工具 (Parallel Image Degradation Synthesis)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例 (Usage Examples):

1. 使用所有可用 CPU 核心平行處理:
   python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./degraded_data

2. 指定使用 4 個工作處理器:
   python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./degraded_data --num_workers 4

3. 測試 10 張影像:
   python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./test_output --max_images 10

注意: 
- 平行處理可以加快全局模糊和低光照的處理速度（約 2-4 倍）
- 物體模糊仍然是循序處理（YOLOv8 在多處理中可能有問題）
- 適合 CPU 處理器核心數 ≥ 4 的電腦
        """
    )
    
    parser.add_argument('--split_dir', type=str, required=True,
                       help='資料分割目錄路徑')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='輸出目錄路徑')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='工作處理器數量（預設：CPU 核心數 - 1）')
    parser.add_argument('--max_images', type=int, default=None,
                       help='每個分割最大處理影像數（用於測試）')
    parser.add_argument('--no_global_blur', action='store_true',
                       help='停用全局運動模糊')
    parser.add_argument('--no_object_blur', action='store_true',
                       help='停用物體運動模糊')
    parser.add_argument('--no_low_light', action='store_true',
                       help='停用低光照降質')
    
    args = parser.parse_args()
    
    # 創建平行化降質流程
    pipeline = ParallelDegradationPipeline(
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
    
    # 處理資料集
    pipeline.process_dataset(
        split_dir=args.split_dir,
        enable_global_blur=not args.no_global_blur,
        enable_object_blur=not args.no_object_blur,
        enable_low_light=not args.no_low_light,
        max_images=args.max_images
    )
    
    print("\n✓ 平行化降質合成完成！")


if __name__ == '__main__':
    main()



