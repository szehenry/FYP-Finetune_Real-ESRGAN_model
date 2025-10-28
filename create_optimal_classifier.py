#!/usr/bin/env python3
"""
创建最优分类器 - 基于数据分析的平衡方法
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from auto_image_curator import AutoImageCurator

def extract_features(image_path):
    """提取图像特征"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. 拉普拉斯方差
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # 3. 纹理分析
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))
    texture_score = np.mean(diff_h) + np.mean(diff_v)
    
    # 4. 局部方差
    block_size = 32
    local_vars = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            local_vars.append(np.var(block))
    
    local_var_mean = np.mean(local_vars) if local_vars else 0
    local_var_std = np.std(local_vars) if local_vars else 0
    
    # 5. 频域分析
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    high_freq_mask = ((y - center_h)**2 + (x - center_w)**2) > (min(h, w) // 6)**2
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # 6. 图像统计
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    return {
        'laplacian_var': laplacian_var,
        'edge_density': edge_density,
        'texture_score': texture_score,
        'local_var_mean': local_var_mean,
        'local_var_std': local_var_std,
        'high_freq_ratio': high_freq_ratio,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity
    }

def analyze_and_create_classifier():
    """分析数据并创建最优分类器"""
    
    image_folder = "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images"
    
    curator = AutoImageCurator(
        root_folder=image_folder,
        csv_path="temp.csv",
        use_ocr=False,
        quality_threshold=0.5
    )
    
    all_images = curator._scan_images()
    
    # 收集特征数据
    blurry_images = [img for img in all_images if '_blurry' in img.name][:150]
    normal_images = [img for img in all_images if '_blurry' not in img.name and '_dark' not in img.name][:150]
    
    print("=== 特征提取和分析 ===")
    print(f"模糊图片: {len(blurry_images)}")
    print(f"正常图片: {len(normal_images)}")
    
    # 提取特征
    features_data = []
    
    print("提取模糊图片特征...")
    for i, img_path in enumerate(blurry_images):
        if i % 30 == 0:
            print(f"  进度: {i+1}/{len(blurry_images)}")
        
        features = extract_features(img_path)
        if features:
            features['label'] = 1  # 1 = 模糊/低质量
            features['filename'] = img_path.name
            features_data.append(features)
    
    print("提取正常图片特征...")
    for i, img_path in enumerate(normal_images):
        if i % 30 == 0:
            print(f"  进度: {i+1}/{len(normal_images)}")
        
        features = extract_features(img_path)
        if features:
            features['label'] = 0  # 0 = 正常/高质量
            features['filename'] = img_path.name
            features_data.append(features)
    
    # 转换为DataFrame
    df = pd.DataFrame(features_data)
    
    # 保存特征数据
    df.to_csv('/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/features_analysis.csv', index=False)
    
    # 分析特征分布
    print(f"\n=== 特征分析 ===")
    
    blurry_df = df[df['label'] == 1]
    normal_df = df[df['label'] == 0]
    
    feature_names = ['laplacian_var', 'edge_density', 'texture_score', 'local_var_std', 'high_freq_ratio']
    
    thresholds = {}
    
    for feature in feature_names:
        blurry_mean = blurry_df[feature].mean()
        blurry_std = blurry_df[feature].std()
        normal_mean = normal_df[feature].mean()
        normal_std = normal_df[feature].std()
        
        # 计算分离度
        separation = abs(blurry_mean - normal_mean) / (blurry_std + normal_std + 1e-6)
        
        print(f"\n{feature}:")
        print(f"  模糊图片: {blurry_mean:.3f} ± {blurry_std:.3f}")
        print(f"  正常图片: {normal_mean:.3f} ± {normal_std:.3f}")
        print(f"  分离度: {separation:.3f}")
        
        # 计算最优阈值 (使用重叠区域的中点)
        if blurry_mean < normal_mean:
            # 模糊图片的值较小
            threshold = (blurry_mean + blurry_std + normal_mean - normal_std) / 2
            direction = "lower"
        else:
            # 模糊图片的值较大
            threshold = (blurry_mean - blurry_std + normal_mean + normal_std) / 2
            direction = "higher"
        
        thresholds[feature] = {'value': threshold, 'direction': direction}
        print(f"  建议阈值: {threshold:.3f} ({direction} than threshold = blurry)")
    
    # 创建简单的决策树规则
    print(f"\n=== 最优分类规则 ===")
    
    # 找出最有区分力的特征
    best_features = []
    for feature in feature_names:
        blurry_vals = blurry_df[feature].values
        normal_vals = normal_df[feature].values
        
        # 计算ROC-AUC近似值
        combined = np.concatenate([blurry_vals, normal_vals])
        labels = np.concatenate([np.ones(len(blurry_vals)), np.zeros(len(normal_vals))])
        
        # 简单的分离度计算
        threshold = thresholds[feature]['value']
        direction = thresholds[feature]['direction']
        
        if direction == "lower":
            predictions = combined < threshold
        else:
            predictions = combined > threshold
        
        accuracy = np.mean(predictions == labels)
        best_features.append((feature, accuracy, threshold, direction))
    
    # 按准确率排序
    best_features.sort(key=lambda x: x[1], reverse=True)
    
    print("特征重要性排序:")
    for i, (feature, accuracy, threshold, direction) in enumerate(best_features):
        print(f"{i+1}. {feature}: {accuracy:.3f} (阈值: {threshold:.3f}, 方向: {direction})")
    
    # 生成最优分类器代码
    print(f"\n=== 生成分类器代码 ===")
    
    classifier_code = f"""
def optimal_blur_detector(image):
    \"\"\"基于数据分析的最优模糊检测器\"\"\"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 提取关键特征
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))
    texture_score = np.mean(diff_h) + np.mean(diff_v)
    
    # 决策规则 (按重要性排序)
    score = 0
    """
    
    # 添加前3个最重要的特征
    for i, (feature, accuracy, threshold, direction) in enumerate(best_features[:3]):
        weight = accuracy * (4 - i)  # 权重递减
        if direction == "lower":
            classifier_code += f"""
    # {feature} (权重: {weight:.2f})
    if {feature} < {threshold:.3f}:
        score += {weight:.2f}
    """
        else:
            classifier_code += f"""
    # {feature} (权重: {weight:.2f})
    if {feature} > {threshold:.3f}:
        score += {weight:.2f}
    """
    
    classifier_code += f"""
    
    # 判断阈值 (基于权重总和)
    threshold_score = {sum(acc * (4-i) for i, (_, acc, _, _) in enumerate(best_features[:3])) * 0.6:.2f}
    is_blurry = score >= threshold_score
    
    return is_blurry, laplacian_var
    """
    
    print(classifier_code)
    
    # 保存分类器代码到文件
    with open('/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/optimal_classifier.py', 'w') as f:
        f.write(classifier_code)
    
    return thresholds, best_features

if __name__ == "__main__":
    thresholds, best_features = analyze_and_create_classifier()
    print("\n分析完成！最优分类器已生成。")
