#!/usr/bin/env python3
"""
分析所有模糊样本 - 找出真正的模糊图片特征
"""

import cv2
import numpy as np
from pathlib import Path
from auto_image_curator import AutoImageCurator
import pandas as pd

def analyze_image_metrics(image_path):
    """分析单张图像的所有指标"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 拉普拉斯方差
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Tenengrad梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sobelx**2 + sobely**2)
    
    # 3. 图像方差
    variance = np.var(gray)
    
    # 4. 标准差
    std_dev = np.std(gray)
    
    # 5. 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    # 6. FFT高频分析
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # 高频掩码 (远离中心)
    mask_high = ((y - center_h)**2 + (x - center_w)**2) > (min(h, w) // 4)**2
    high_freq_energy = np.sum(magnitude_spectrum[mask_high])
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # 7. 局部方差 - 将图像分块计算方差
    block_size = 32
    local_vars = []
    for i in range(0, gray.shape[0] - block_size, block_size):
        for j in range(0, gray.shape[1] - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            local_vars.append(np.var(block))
    
    local_var_mean = np.mean(local_vars) if local_vars else 0
    local_var_std = np.std(local_vars) if local_vars else 0
    
    return {
        'filename': image_path.name,
        'laplacian_var': laplacian_var,
        'tenengrad': tenengrad,
        'variance': variance,
        'std_dev': std_dev,
        'edge_density': edge_density,
        'high_freq_ratio': high_freq_ratio,
        'local_var_mean': local_var_mean,
        'local_var_std': local_var_std,
        'is_marked_blurry': '_blurry' in image_path.name,
        'is_marked_dark': '_dark' in image_path.name
    }

def main():
    """主分析函数"""
    print("=== 全面分析图像模糊特征 ===")
    
    image_folder = "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images"
    
    curator = AutoImageCurator(
        root_folder=image_folder,
        csv_path="temp.csv",
        use_ocr=False,
        quality_threshold=0.5
    )
    
    all_images = curator._scan_images()
    
    # 分类图片
    blurry_images = [img for img in all_images if '_blurry' in img.name]
    dark_images = [img for img in all_images if '_dark' in img.name]
    normal_images = [img for img in all_images if '_blurry' not in img.name and '_dark' not in img.name]
    
    print(f"找到 {len(blurry_images)} 张标记为模糊的图片")
    print(f"找到 {len(dark_images)} 张标记为暗的图片")
    print(f"找到 {len(normal_images)} 张正常图片")
    
    # 分析样本 - 每类取50张
    sample_size = 50
    test_images = (
        blurry_images[:sample_size] + 
        dark_images[:sample_size] + 
        normal_images[:sample_size]
    )
    
    print(f"\n分析 {len(test_images)} 张图片...")
    
    results = []
    for i, img_path in enumerate(test_images):
        if i % 20 == 0:
            print(f"进度: {i+1}/{len(test_images)}")
        
        metrics = analyze_image_metrics(img_path)
        if metrics:
            results.append(metrics)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果
    df.to_csv('/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/image_analysis_results.csv', index=False)
    
    # 分析统计
    print(f"\n=== 统计分析 ===")
    
    # 按类别分组
    blurry_df = df[df['is_marked_blurry'] == True]
    dark_df = df[df['is_marked_dark'] == True]
    normal_df = df[(df['is_marked_blurry'] == False) & (df['is_marked_dark'] == False)]
    
    print(f"\n模糊图片统计 ({len(blurry_df)} 张):")
    if len(blurry_df) > 0:
        print(f"  拉普拉斯方差: {blurry_df['laplacian_var'].mean():.2f} ± {blurry_df['laplacian_var'].std():.2f}")
        print(f"  Tenengrad: {blurry_df['tenengrad'].mean():.2f} ± {blurry_df['tenengrad'].std():.2f}")
        print(f"  边缘密度: {blurry_df['edge_density'].mean():.4f} ± {blurry_df['edge_density'].std():.4f}")
        print(f"  高频比例: {blurry_df['high_freq_ratio'].mean():.4f} ± {blurry_df['high_freq_ratio'].std():.4f}")
    
    print(f"\n正常图片统计 ({len(normal_df)} 张):")
    if len(normal_df) > 0:
        print(f"  拉普拉斯方差: {normal_df['laplacian_var'].mean():.2f} ± {normal_df['laplacian_var'].std():.2f}")
        print(f"  Tenengrad: {normal_df['tenengrad'].mean():.2f} ± {normal_df['tenengrad'].std():.2f}")
        print(f"  边缘密度: {normal_df['edge_density'].mean():.4f} ± {normal_df['edge_density'].std():.4f}")
        print(f"  高频比例: {normal_df['high_freq_ratio'].mean():.4f} ± {normal_df['high_freq_ratio'].std():.4f}")
    
    # 找出最能区分的指标
    print(f"\n=== 区分能力分析 ===")
    
    if len(blurry_df) > 0 and len(normal_df) > 0:
        metrics_to_check = ['laplacian_var', 'tenengrad', 'edge_density', 'high_freq_ratio', 'local_var_mean']
        
        for metric in metrics_to_check:
            blurry_mean = blurry_df[metric].mean()
            normal_mean = normal_df[metric].mean()
            
            # 计算分离度 (两组均值差除以标准差之和)
            blurry_std = blurry_df[metric].std()
            normal_std = normal_df[metric].std()
            separation = abs(blurry_mean - normal_mean) / (blurry_std + normal_std + 1e-6)
            
            print(f"{metric}: 分离度 = {separation:.3f}")
            print(f"  模糊图片: {blurry_mean:.3f}")
            print(f"  正常图片: {normal_mean:.3f}")
            
            # 建议阈值
            if blurry_mean < normal_mean:
                threshold = (blurry_mean + blurry_std + normal_mean - normal_std) / 2
                print(f"  建议阈值: {threshold:.3f} (低于此值为模糊)")
            else:
                threshold = (blurry_mean - blurry_std + normal_mean + normal_std) / 2
                print(f"  建议阈值: {threshold:.3f} (高于此值为模糊)")
            print()
    
    # 找出最模糊和最清晰的图片
    print(f"\n=== 极值分析 ===")
    
    # 按拉普拉斯方差排序
    df_sorted = df.sort_values('laplacian_var')
    
    print("拉普拉斯方差最低的10张图片 (可能最模糊):")
    for i in range(min(10, len(df_sorted))):
        row = df_sorted.iloc[i]
        print(f"  {row['filename']}: {row['laplacian_var']:.2f} (标记: {'模糊' if row['is_marked_blurry'] else '暗' if row['is_marked_dark'] else '正常'})")
    
    print("\n拉普拉斯方差最高的10张图片 (可能最清晰):")
    for i in range(max(0, len(df_sorted)-10), len(df_sorted)):
        row = df_sorted.iloc[i]
        print(f"  {row['filename']}: {row['laplacian_var']:.2f} (标记: {'模糊' if row['is_marked_blurry'] else '暗' if row['is_marked_dark'] else '正常'})")

if __name__ == "__main__":
    main()
