#!/usr/bin/env python3
"""
自动图像筛选工具测试脚本
用于验证系统功能和准确性
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from auto_image_curator import AutoImageCurator

def create_test_images(test_folder: Path):
    """创建测试图像"""
    test_folder.mkdir(parents=True, exist_ok=True)
    
    print("创建测试图像...")
    
    # 1. 高质量清晰图像
    clear_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    # 添加一些结构化内容
    cv2.rectangle(clear_img, (100, 100), (300, 200), (100, 100, 100), -1)  # 建筑物
    cv2.rectangle(clear_img, (400, 300), (500, 400), (0, 100, 200), -1)    # 车辆
    cv2.imwrite(str(test_folder / "clear_building_vehicle.jpg"), clear_img)
    
    # 2. 模糊图像
    blur_img = np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
    blur_img = cv2.GaussianBlur(blur_img, (15, 15), 0)  # 添加模糊
    cv2.imwrite(str(test_folder / "blurry_image.jpg"), blur_img)
    
    # 3. 低光照图像
    dark_img = np.random.randint(10, 50, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_folder / "dark_image.jpg"), dark_img)
    
    # 4. 过曝图像
    bright_img = np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_folder / "overexposed_image.jpg"), bright_img)
    
    # 5. 夜间但质量好的图像
    night_img = np.random.randint(30, 80, (480, 640, 3), dtype=np.uint8)
    # 添加一些亮点（路灯等）
    cv2.circle(night_img, (200, 150), 20, (200, 200, 150), -1)
    cv2.circle(night_img, (400, 200), 15, (180, 180, 120), -1)
    cv2.imwrite(str(test_folder / "night_good_quality.jpg"), night_img)
    
    # 6. 包含绿色植被的图像
    vegetation_img = np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
    # 添加绿色区域
    vegetation_img[100:300, 200:500] = [50, 150, 50]  # 绿色植被
    cv2.imwrite(str(test_folder / "vegetation_scene.jpg"), vegetation_img)
    
    print(f"测试图像已创建在: {test_folder}")

def run_basic_test():
    """运行基础功能测试"""
    print("=== 基础功能测试 ===")
    
    # 创建测试文件夹
    test_folder = Path("test_images_auto")
    create_test_images(test_folder)
    
    # 创建筛选器
    csv_path = test_folder / "test_results.csv"
    curator = AutoImageCurator(
        root_folder=str(test_folder),
        csv_path=str(csv_path),
        use_ocr=False,  # 关闭OCR以加快测试
        quality_threshold=0.5
    )
    
    # 处理图像
    print("\n开始处理测试图像...")
    df = curator.process_folder()
    
    # 显示结果
    curator.print_statistics()
    
    if not df.empty:
        print(f"\n=== 测试结果 ===")
        for _, row in df.iterrows():
            filename = Path(row['filepath']).name
            print(f"{filename}: 场景={row['scene_type']}, 质量={row['quality_score']:.3f}, "
                  f"夜间={'是' if row['is_night'] else '否'}")
    
    # 检查拒绝文件夹
    rejected_folder = test_folder / "rejected_images"
    if rejected_folder.exists():
        rejected_files = list(rejected_folder.glob("*.jpg"))
        print(f"\n拒绝的图像 ({len(rejected_files)} 个):")
        for f in rejected_files:
            print(f"  {f.name}")
    
    return test_folder

def test_individual_functions():
    """测试各个功能模块"""
    print("\n=== 功能模块测试 ===")
    
    # 创建一个简单的测试图像
    test_img = np.random.randint(100, 150, (300, 400, 3), dtype=np.uint8)
    
    # 创建筛选器实例
    curator = AutoImageCurator(
        root_folder=".",
        csv_path="temp.csv",
        use_ocr=False,
        quality_threshold=0.5
    )
    
    # 测试模糊检测
    print("测试模糊检测...")
    is_blurry, blur_score = curator._detect_blur(test_img)
    print(f"  模糊检测: {'是' if is_blurry else '否'}, 分数: {blur_score:.2f}")
    
    # 测试光照分析
    print("测试光照分析...")
    lighting_info = curator._analyze_lighting(test_img)
    print(f"  平均亮度: {lighting_info['mean_brightness']:.2f}")
    print(f"  是否夜间: {'是' if lighting_info['is_night'] else '否'}")
    print(f"  光照质量: {lighting_info['lighting_quality']}")
    
    # 测试场景特征检测
    print("测试场景特征检测...")
    scene_features = curator._detect_scene_features(test_img)
    print(f"  检测到的场景: {scene_features}")
    
    # 测试对象检测（OpenCV）
    print("测试OpenCV对象检测...")
    opencv_objects = curator._detect_objects_opencv(test_img)
    print(f"  检测到的对象: {opencv_objects}")

def main():
    """主测试函数"""
    print("自动图像筛选工具测试")
    print("=" * 40)
    
    try:
        # 运行功能模块测试
        test_individual_functions()
        
        # 运行基础测试
        test_folder = run_basic_test()
        
        print(f"\n=== 测试完成 ===")
        print(f"测试文件夹: {test_folder}")
        print("请检查生成的CSV文件和拒绝文件夹中的结果")
        
        # 清理选项
        response = input("\n是否删除测试文件夹? (y/N): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(test_folder)
            print("测试文件夹已删除")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
