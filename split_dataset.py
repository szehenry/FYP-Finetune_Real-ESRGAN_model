#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割脚本 (Dataset Splitting Script)
用途：将增强后的影像分割为 train/val/test 集合

策略：Stratified Random Split
- 按原始影像分组（避免数据泄露）
- 每个数据集按相同比例分割（确保多样性）
- 使用固定 random seed（确保可重现）

作者：FYP Project
日期：2025-10-29
"""

import os
import random
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

# ==================== 配置区 ====================

# 数据集路径
FYP_DIR = Path(r"C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP")
IMAGES_BASE_DIR = Path(r"C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP_Images")

# 数据集文件夹
DATASET_FOLDERS = {
    "Aerial-Traffic": "Images_Aerial-Traffic",
    "FloodNet": "Images_FloodNet",
    "UAV-VisLoc": "Images_UAV-VisLoc",
    "VisDrone2019": "Images_VisDrone2019",
    "SwissOkutama": "Images_SwissOkutama"
}

# 分割比例
SPLIT_RATIOS = {
    "train": 0.70,    # 70%
    "val": 0.15,      # 15%
    "test": 0.15      # 15%
}

# Random Seed（用于可重现性）
RANDOM_SEED = 42

# 输出目录
OUTPUT_DIR = FYP_DIR / "own_Windows_data_split_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 增强影像的识别后缀
AUGMENTATION_SUFFIXES = ['_flip_h', '_flip_v', '_rot90', '_rot180', '_rot270', '_crop']


# ==================== 核心函数 ====================

def is_original_image(filename: str) -> bool:
    """判断是否为原始影像（非增强版本）"""
    stem = Path(filename).stem
    for suffix in AUGMENTATION_SUFFIXES:
        if suffix in stem:
            return False
    return True


def get_original_name(filename: str) -> str:
    """获取增强影像对应的原始影像名称"""
    stem = Path(filename).stem
    ext = Path(filename).suffix
    
    # 移除所有增强后缀
    for suffix in AUGMENTATION_SUFFIXES:
        if suffix in stem:
            # 处理 crop0, crop1 等
            if suffix == '_crop':
                # 移除 _crop0, _crop1, _crop2 等
                import re
                stem = re.sub(r'_crop\d+$', '', stem)
            else:
                stem = stem.replace(suffix, '')
    
    return stem + ext


def collect_image_groups(dataset_folder: Path) -> Dict[str, List[str]]:
    """
    收集影像分组
    返回: {原始影像路径: [增强影像路径列表]}
    """
    if not dataset_folder.exists():
        print(f"  ⚠️  资料夹不存在: {dataset_folder}")
        return {}
    
    # 获取所有影像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(dataset_folder.glob(f'*{ext}'))
    
    # 去重（Windows 系统不区分大小写，会导致重复）
    image_files = list(set(image_files))
    
    # 按文件名排序以确保跨平台可重现性（不是按完整路径）
    image_files = sorted(image_files, key=lambda x: x.name)
    
    # 排除 rejected_images 文件夹
    image_files = [f for f in image_files if 'rejected_images' not in str(f)]
    
    # 分组
    groups = defaultdict(list)
    original_images = set()
    
    for img_path in image_files:
        if is_original_image(img_path.name):
            original_images.add(str(img_path))
        else:
            # 找到对应的原始影像
            original_name = get_original_name(img_path.name)
            original_path = img_path.parent / original_name
            if original_path.exists():
                groups[str(original_path)].append(str(img_path))
    
    # 为每个原始影像创建完整的分组（包括自己）
    complete_groups = {}
    for original_path in original_images:
        complete_groups[original_path] = [original_path] + groups.get(original_path, [])
    
    return complete_groups


def stratified_split(groups: Dict[str, List[str]], 
                     ratios: Dict[str, float],
                     seed: int = 42) -> Dict[str, List[str]]:
    """
    执行 Stratified Random Split
    
    参数:
        groups: {原始影像: [所有相关影像]}
        ratios: {"train": 0.70, "val": 0.15, "test": 0.15}
        seed: random seed
    
    返回:
        {"train": [所有训练影像], "val": [...], "test": [...]}
    """
    random.seed(seed)
    
    # 获取所有原始影像并随机打乱
    original_images = list(groups.keys())
    random.shuffle(original_images)
    
    # 计算分割点
    total = len(original_images)
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])
    
    # 分割原始影像
    train_originals = original_images[:train_end]
    val_originals = original_images[train_end:val_end]
    test_originals = original_images[val_end:]
    
    # 收集所有相关影像（包括增强版本）
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for original in train_originals:
        splits["train"].extend(groups[original])
    
    for original in val_originals:
        splits["val"].extend(groups[original])
    
    for original in test_originals:
        splits["test"].extend(groups[original])
    
    return splits


def save_split_lists(splits: Dict[str, List[str]], output_dir: Path):
    """保存分割清单到文件"""
    for split_name, image_list in splits.items():
        output_file = output_dir / f"{split_name}_list.txt"
        with open(output_file, 'w') as f:
            for img_path in sorted(image_list):
                f.write(f"{img_path}\n")
        print(f"  ✓ {split_name:5s}: {len(image_list):4d} 张 → {output_file.name}")


def generate_statistics(all_splits: Dict[str, Dict[str, List[str]]], 
                        output_dir: Path):
    """生成统计报告"""
    
    # 整体统计
    total_stats = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    
    for split in ["train", "val", "test"]:
        for dataset_name in all_splits:
            total_stats[split] += len(all_splits[dataset_name][split])
    
    # 详细统计
    detailed_stats = {}
    for dataset_name, splits in all_splits.items():
        detailed_stats[dataset_name] = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        }
    
    # 保存 JSON
    stats = {
        "split_ratios": SPLIT_RATIOS,
        "random_seed": RANDOM_SEED,
        "total_statistics": total_stats,
        "per_dataset_statistics": detailed_stats
    }
    
    stats_file = output_dir / "split_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 统计报告已保存: {stats_file}")
    
    return stats


def print_visualization(stats: Dict):
    """打印可视化分布"""
    print("\n" + "=" * 80)
    print("📊 数据集分割统计")
    print("=" * 80)
    
    print(f"\n🎲 Random Seed: {stats['random_seed']}")
    print(f"📐 分割比例: Train {stats['split_ratios']['train']:.0%} / "
          f"Val {stats['split_ratios']['val']:.0%} / "
          f"Test {stats['split_ratios']['test']:.0%}\n")
    
    # 表头
    print(f"{'Dataset':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    # 每个数据集
    for dataset_name, ds_stats in stats['per_dataset_statistics'].items():
        print(f"{dataset_name:<20} "
              f"{ds_stats['train']:>8} "
              f"{ds_stats['val']:>8} "
              f"{ds_stats['test']:>8} "
              f"{ds_stats['total']:>8}")
    
    print("-" * 60)
    
    # 总计
    total = stats['total_statistics']
    total_sum = total['train'] + total['val'] + total['test']
    print(f"{'TOTAL':<20} "
          f"{total['train']:>8} "
          f"{total['val']:>8} "
          f"{total['test']:>8} "
          f"{total_sum:>8}")
    
    # 百分比验证
    print("\n📊 实际比例:")
    print(f"  Train: {total['train']/total_sum*100:.1f}%")
    print(f"  Val:   {total['val']/total_sum*100:.1f}%")
    print(f"  Test:  {total['test']/total_sum*100:.1f}%")


# ==================== 主程序 ====================

def main():
    """主函数"""
    print("=" * 80)
    print("🎲 数据集分割开始 (Stratified Random Split)")
    print("=" * 80)
    print(f"\n⚙️  配置:")
    print(f"  分割比例: {SPLIT_RATIOS['train']:.0%} / {SPLIT_RATIOS['val']:.0%} / {SPLIT_RATIOS['test']:.0%}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  策略: Stratified Split (按数据集分层)")
    print()
    
    all_splits = {}
    
    # 对每个数据集执行分割
    for dataset_name, folder_name in DATASET_FOLDERS.items():
        print(f"▶ 处理数据集: {dataset_name}")
        
        dataset_folder = IMAGES_BASE_DIR / folder_name
        
        # 1. 收集影像分组
        groups = collect_image_groups(dataset_folder)
        
        if not groups:
            print(f"  ⚠️  没有找到影像，跳过")
            continue
        
        print(f"  原始影像: {len(groups)} 张")
        total_images = sum(len(img_list) for img_list in groups.values())
        print(f"  总影像数（含增强）: {total_images} 张")
        
        # 2. 执行分割
        splits = stratified_split(groups, SPLIT_RATIOS, RANDOM_SEED)
        all_splits[dataset_name] = splits
        
        print(f"  ✓ Train: {len(splits['train'])} 张")
        print(f"  ✓ Val:   {len(splits['val'])} 张")
        print(f"  ✓ Test:  {len(splits['test'])} 张")
        print()
    
    # 3. 合并所有数据集的分割结果
    print("=" * 80)
    print("💾 保存分割清单")
    print("=" * 80)
    
    merged_splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for dataset_name, splits in all_splits.items():
        for split_name in ["train", "val", "test"]:
            merged_splits[split_name].extend(splits[split_name])
    
    # 4. 保存文件清单
    save_split_lists(merged_splits, OUTPUT_DIR)
    
    # 5. 生成统计报告
    stats = generate_statistics(all_splits, OUTPUT_DIR)
    
    # 6. 打印可视化
    print_visualization(stats)
    
    print("\n" + "=" * 80)
    print("🎉 数据集分割完成！")
    print("=" * 80)
    print(f"\n📁 所有结果已保存至: {OUTPUT_DIR}")
    print(f"\n📄 生成的文件:")
    print(f"  - train_list.txt: 训练集影像清单")
    print(f"  - val_list.txt: 验证集影像清单")
    print(f"  - test_list.txt: 测试集影像清单")
    print(f"  - split_statistics.json: 详细统计信息")
    print("\n⚠️  重要提醒:")
    print("  - 这些清单包含完整的绝对路径")
    print("  - 原始影像及其所有增强版本都在同一集合中")
    print("  - 使用相同的 random seed 可以重现相同的分割")
    print("  - 训练时请使用这些清单加载影像")
    print()


if __name__ == "__main__":
    main()

