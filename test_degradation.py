"""
降質合成測試腳本 (Degradation Synthesis Test Script)
用於快速測試單張影像的降質效果

使用方法 (Usage):
    python test_degradation.py --image path/to/image.jpg --output ./test_output
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
import sys

# 導入降質合成器
from degradation_synthesis import (
    MotionBlurSynthesizer,
    ObjectMotionBlurSynthesizer,
    LowLightSynthesizer
)


def create_comparison_grid(original, degraded_images, titles, output_path):
    """
    創建對比網格圖
    
    Args:
        original: 原始影像
        degraded_images: 降質影像列表
        titles: 標題列表
        output_path: 輸出路徑
    """
    # 調整所有影像到相同高度
    target_height = 480
    
    def resize_keep_aspect(img, target_h):
        h, w = img.shape[:2]
        ratio = target_h / h
        new_w = int(w * ratio)
        return cv2.resize(img, (new_w, target_h))
    
    original_resized = resize_keep_aspect(original, target_height)
    degraded_resized = [resize_keep_aspect(img, target_height) for img in degraded_images]
    
    # 添加標題
    def add_title(img, title):
        # 創建標題欄
        title_bar = np.ones((40, img.shape[1], 3), dtype=np.uint8) * 255
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 計算文字大小和位置
        (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
        x = (title_bar.shape[1] - text_w) // 2
        y = (title_bar.shape[0] + text_h) // 2
        
        cv2.putText(title_bar, title, (x, y), font, font_scale, (0, 0, 0), thickness)
        
        # 合併標題和影像
        return np.vstack([title_bar, img])
    
    original_titled = add_title(original_resized, "Original (原始)")
    degraded_titled = [add_title(img, title) for img, title in zip(degraded_resized, titles)]
    
    # 創建網格
    # 第一行：原始影像
    # 第二行：降質影像們
    row1 = original_titled
    row2 = np.hstack(degraded_titled)
    
    # 調整 row1 寬度以匹配 row2
    if row1.shape[1] < row2.shape[1]:
        padding = np.ones((row1.shape[0], row2.shape[1] - row1.shape[1], 3), dtype=np.uint8) * 255
        row1 = np.hstack([row1, padding])
    
    grid = np.vstack([row1, row2])
    
    # 儲存
    cv2.imwrite(str(output_path), grid)
    print(f"對比圖已儲存到: {output_path}")
    
    return grid


def main():
    parser = argparse.ArgumentParser(description='測試影像降質效果')
    parser.add_argument('--image', type=str, required=True,
                       help='輸入影像路徑')
    parser.add_argument('--output', type=str, default='./test_degradation_output',
                       help='輸出目錄')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='計算設備')
    parser.add_argument('--show', action='store_true',
                       help='顯示結果視窗')
    
    args = parser.parse_args()
    
    # 檢查影像是否存在
    if not Path(args.image).exists():
        print(f"錯誤: 影像不存在 - {args.image}")
        sys.exit(1)
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 讀取影像
    print(f"讀取影像: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"錯誤: 無法讀取影像 - {args.image}")
        sys.exit(1)
    
    print(f"影像尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 設定設備
    import torch
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n使用設備: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 初始化合成器
    print("\n初始化降質合成器...")
    global_blur = MotionBlurSynthesizer(device=device)
    object_blur = ObjectMotionBlurSynthesizer(device=device)
    low_light = LowLightSynthesizer(device=device)
    
    degraded_images = []
    titles = []
    metadata_list = []
    
    # 1. 全局運動模糊
    print("\n[1/3] 生成全局運動模糊...")
    try:
        deg_global, params_global = global_blur.synthesize(image, {})
        degraded_images.append(deg_global)
        titles.append("Global Blur")
        metadata_list.append(params_global)
        
        # 儲存
        cv2.imwrite(str(output_dir / 'global_blur.jpg'), deg_global)
        with open(output_dir / 'global_blur.json', 'w', encoding='utf-8') as f:
            json.dump(params_global, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 全局模糊完成")
        print(f"  - 角度: {params_global['angle']:.1f}°")
        print(f"  - 運動長度: {params_global['motion_length']:.1f} 像素")
    except Exception as e:
        print(f"✗ 全局模糊失敗: {e}")
    
    # 2. 物體運動模糊
    print("\n[2/3] 生成物體運動模糊...")
    try:
        deg_object, params_object = object_blur.synthesize(image, {})
        
        if params_object.get('objects_detected', 0) > 0:
            degraded_images.append(deg_object)
            titles.append("Object Blur")
            metadata_list.append(params_object)
            
            # 儲存
            cv2.imwrite(str(output_dir / 'object_blur.jpg'), deg_object)
            with open(output_dir / 'object_blur.json', 'w', encoding='utf-8') as f:
                json.dump(params_object, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 物體模糊完成")
            print(f"  - 偵測到物體數: {params_object['objects_detected']}")
            print(f"  - 物體類別: {', '.join(params_object['detected_classes'])}")
        else:
            print(f"⚠ 未偵測到物體，跳過物體模糊")
    except Exception as e:
        print(f"✗ 物體模糊失敗: {e}")
    
    # 3. 低光照降質
    print("\n[3/3] 生成低光照降質...")
    try:
        deg_lowlight, params_lowlight = low_light.synthesize(image, {})
        degraded_images.append(deg_lowlight)
        titles.append("Low-light")
        metadata_list.append(params_lowlight)
        
        # 儲存
        cv2.imwrite(str(output_dir / 'low_light.jpg'), deg_lowlight)
        with open(output_dir / 'low_light.json', 'w', encoding='utf-8') as f:
            json.dump(params_lowlight, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 低光照完成")
        print(f"  - 曝光因子: {params_lowlight['exposure_factor']:.2f}")
        print(f"  - 讀取噪聲: {params_lowlight['read_noise_std']:.1f}")
    except Exception as e:
        print(f"✗ 低光照失敗: {e}")
    
    # 創建對比網格圖
    if len(degraded_images) > 0:
        print("\n創建對比圖...")
        grid = create_comparison_grid(
            image, 
            degraded_images, 
            titles,
            output_dir / 'comparison_grid.jpg'
        )
        
        # 顯示
        if args.show:
            # 調整視窗大小以適應螢幕
            cv2.namedWindow('Degradation Comparison', cv2.WINDOW_NORMAL)
            cv2.imshow('Degradation Comparison', grid)
            print("\n按任意鍵關閉視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # 儲存完整摘要
    summary = {
        'input_image': args.image,
        'image_size': f"{image.shape[1]}x{image.shape[0]}",
        'device': device,
        'degradations_generated': len(degraded_images),
        'metadata': metadata_list
    }
    
    with open(output_dir / 'test_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 測試完成！")
    print(f"輸出目錄: {output_dir}")
    print(f"生成降質影像數: {len(degraded_images)}")


if __name__ == '__main__':
    main()

