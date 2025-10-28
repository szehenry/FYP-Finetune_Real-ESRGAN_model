#!/usr/bin/env python3
"""
自动图像筛选工具使用示例
演示如何使用AutoImageCurator进行批量图像处理
"""

from pathlib import Path
from auto_image_curator import AutoImageCurator

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 设置路径（请根据实际情况修改）
    image_folder = "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images"
    csv_output = "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/auto_results.csv"
    
    # 检查文件夹是否存在
    if not Path(image_folder).exists():
        print(f"图像文件夹不存在: {image_folder}")
        print("请修改 image_folder 变量为你的图像文件夹路径")
        return
    
    # 创建筛选器
    curator = AutoImageCurator(
        root_folder=image_folder,
        csv_path=csv_output,
        use_ocr=True,  # 启用OCR文本检测
        quality_threshold=0.6  # 质量阈值
    )
    
    # 处理图像
    print("开始处理图像...")
    df = curator.process_folder()
    
    # 显示结果
    curator.print_statistics()
    
    if not df.empty:
        print(f"\n处理完成！结果已保存到: {csv_output}")
        print(f"接受的图像数量: {len(df)}")
        
        # 显示前几条记录
        print("\n前5条记录预览:")
        print(df.head().to_string())
    else:
        print("没有图像被接受")

def example_multiple_folders():
    """处理多个文件夹的示例"""
    print("\n=== 处理多个文件夹示例 ===")
    
    # 定义多个图像文件夹
    folders = [
        "/path/to/dataset1",
        "/path/to/dataset2", 
        "/path/to/dataset3"
    ]
    
    for i, folder in enumerate(folders, 1):
        print(f"\n处理文件夹 {i}/{len(folders)}: {folder}")
        
        if not Path(folder).exists():
            print(f"跳过不存在的文件夹: {folder}")
            continue
        
        # 为每个文件夹创建单独的CSV文件
        csv_output = Path(folder) / "auto_results.csv"
        
        curator = AutoImageCurator(
            root_folder=folder,
            csv_path=str(csv_output),
            use_ocr=False,  # 为了速度，禁用OCR
            quality_threshold=0.7  # 使用更严格的标准
        )
        
        df = curator.process_folder()
        curator.print_statistics()

def example_custom_settings():
    """自定义设置示例"""
    print("\n=== 自定义设置示例 ===")
    
    image_folder = "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images"
    
    if not Path(image_folder).exists():
        print(f"图像文件夹不存在: {image_folder}")
        return
    
    # 高质量模式 - 严格筛选
    print("高质量模式（严格筛选）:")
    curator_strict = AutoImageCurator(
        root_folder=image_folder,
        csv_path=str(Path(image_folder) / "high_quality_results.csv"),
        rejected_folder=str(Path(image_folder) / "rejected_strict"),
        use_ocr=True,
        quality_threshold=0.8  # 高阈值
    )
    
    # 快速模式 - 宽松筛选
    print("\n快速模式（宽松筛选）:")
    curator_fast = AutoImageCurator(
        root_folder=image_folder,
        csv_path=str(Path(image_folder) / "fast_results.csv"),
        rejected_folder=str(Path(image_folder) / "rejected_fast"),
        use_ocr=False,  # 禁用OCR加快速度
        quality_threshold=0.4  # 低阈值
    )

def main():
    """主函数"""
    print("自动图像筛选工具使用示例")
    print("=" * 50)
    
    try:
        # 运行基础示例
        example_basic_usage()
        
        # 其他示例（注释掉以避免实际运行）
        # example_multiple_folders()
        # example_custom_settings()
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
