#!/usr/bin/env python3
"""
自动图像筛选和标注工具
用于Real-ESRGAN项目的数据集自动化准备

功能：
- 自动检测图像质量（运动模糊、亮度）
- 自动识别对象（道路、建筑、车辆、人物等）
- 自动检测文本和白天/夜间场景
- 自动保存高质量图像信息到CSV
- 自动移动低质量图像到拒绝文件夹

作者：为FYP项目创建
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
import shutil
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# 可选依赖
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("提示：PaddleOCR未安装，OCR功能不可用。运行 'pip install paddleocr' 来启用。")

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("提示：PyTorch未安装，将使用基础检测方法。运行 'pip install torch torchvision' 来启用高级功能。")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("提示：YOLOv8未安装，将使用OpenCV级联分类器。运行 'pip install ultralytics' 来启用YOLO检测。")

class AutoImageCurator:
    def __init__(self, root_folder: str, csv_path: str, rejected_folder: str = None, 
                 use_ocr: bool = True, quality_threshold: float = 0.6):
        self.root_folder = Path(root_folder)
        self.csv_path = Path(csv_path)
        self.quality_threshold = quality_threshold
        
        # 设置拒绝文件夹
        if rejected_folder:
            self.rejected_folder = Path(rejected_folder)
        else:
            self.rejected_folder = self.root_folder / "rejected_images"
        
        self.rejected_folder.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 场景类型映射
        self.scene_types = {
            'road': ['road', 'street', 'highway', 'path', 'sidewalk'],
            'building': ['building', 'house', 'skyscraper', 'structure'],
            'infrastructure': ['bridge', 'tunnel', 'tower', 'pole'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'vegetation': ['tree', 'grass', 'plant', 'forest'],
            'person_animal': ['person', 'people', 'dog', 'cat', 'bird', 'animal'],
            'other': []
        }
        
        # 初始化检测器
        self._init_detectors(use_ocr)
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'accepted': 0,
            'rejected': 0,
            'blur_rejected': 0,
            'low_light_rejected': 0,
            'poor_quality_rejected': 0
        }
        
        print(f"自动图像筛选器初始化完成")
        print(f"输入文件夹: {self.root_folder}")
        print(f"CSV输出: {self.csv_path}")
        print(f"拒绝文件夹: {self.rejected_folder}")
    
    def _init_detectors(self, use_ocr: bool):
        """初始化各种检测器"""
        # OCR初始化
        self.use_ocr = use_ocr and OCR_AVAILABLE
        if self.use_ocr:
            print("正在初始化OCR引擎...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            print("OCR引擎初始化完成")
        
        # YOLO初始化
        self.use_yolo = YOLO_AVAILABLE
        if self.use_yolo:
            try:
                print("正在初始化YOLO检测器...")
                self.yolo_model = YOLO('yolov8n.pt')  # 使用nano版本，速度快
                print("YOLO检测器初始化完成")
            except Exception as e:
                print(f"YOLO初始化失败: {e}")
                self.use_yolo = False
        
        # OpenCV级联分类器作为备选
        self.use_cascade = True
        try:
            # 人脸检测器
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # 车辆检测器（如果可用）
            car_cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
            if os.path.exists(car_cascade_path):
                self.car_cascade = cv2.CascadeClassifier(car_cascade_path)
            else:
                self.car_cascade = None
        except Exception as e:
            print(f"OpenCV级联分类器初始化失败: {e}")
            self.use_cascade = False
    
    def _scan_images(self) -> List[Path]:
        """扫描图像文件"""
        image_files = []
        for ext in self.image_extensions:
            pattern = f"**/*{ext}"
            image_files.extend(self.root_folder.glob(pattern))
            # 也搜索大写扩展名
            pattern = f"**/*{ext.upper()}"
            image_files.extend(self.root_folder.glob(pattern))
        
        # 排序并去重
        image_files = list(set(image_files))
        image_files.sort()
        return image_files
    
    def _detect_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """智能图像质量检测 - 多维度综合评估"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. 拉普拉斯方差 (清晰度主要指标)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. 边缘密度和质量
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # 3. 局部方差分析
        block_size = 32
        local_vars = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                local_vars.append(np.var(block))
        
        local_var_mean = np.mean(local_vars) if local_vars else 0
        local_var_std = np.std(local_vars) if local_vars else 0
        
        # 4. 频域分析
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 高频能量比例
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        high_freq_mask = ((y - center_h)**2 + (x - center_w)**2) > (min(h, w) // 6)**2
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # 5. 纹理分析 - 灰度共生矩阵的简化版本
        # 计算水平和垂直方向的灰度差异
        diff_h = np.abs(np.diff(gray, axis=1))
        diff_v = np.abs(np.diff(gray, axis=0))
        texture_score = np.mean(diff_h) + np.mean(diff_v)
        
        # 基于数据分析的最优模糊检测器
        # 决策规则 (按重要性排序)
        score = 0
        
        # texture_score (权重: 2.55) - 最重要的特征
        if texture_score > 198.429:
            score += 2.55
        
        # laplacian_var (权重: 1.80) - 第二重要
        if laplacian_var > 866.382:
            score += 1.80
        
        # high_freq_ratio (权重: 1.03) - 第三重要
        if high_freq_ratio > 0.659:
            score += 1.03
        
        # 调整后的判断阈值 (提高准确率)
        threshold_score = 2.5  # 降低阈值以减少误检正常图片
        is_poor_quality = score >= threshold_score
        
        # 计算综合质量分数
        quality_score = laplacian_var + texture_score * 5 + high_freq_ratio * 1000
        
        return is_poor_quality, quality_score
    
    def _analyze_lighting(self, image: np.ndarray) -> Dict[str, any]:
        """分析光照条件"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 基本亮度统计
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # 直方图分析
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        
        # 检测低光照（大部分像素在低亮度区域）
        low_light_ratio = np.sum(hist_norm[:50])  # 0-49亮度范围
        dark_ratio = np.sum(hist_norm[:30])       # 0-29亮度范围
        
        # 检测过曝（大部分像素在高亮度区域）
        overexposed_ratio = np.sum(hist_norm[200:])  # 200-255亮度范围
        
        # 判断是否为夜间
        is_night = (mean_brightness < 80 and low_light_ratio > 0.6) or dark_ratio > 0.4
        
        # 判断是否为低质量光照 - 更宽松的标准
        is_too_dark = dark_ratio > 0.85 or mean_brightness < 20  # 更宽松的暗度标准
        is_overexposed = overexposed_ratio > 0.4 and mean_brightness > 220  # 更宽松的过曝标准
        
        # 检查阴影区域的细节可见性
        shadow_detail_visible = True
        if low_light_ratio > 0.3:  # 有较多阴影区域
            # 检查阴影区域的纹理
            shadow_mask = gray < 80
            if np.sum(shadow_mask) > 0:
                shadow_region = gray[shadow_mask]
                shadow_std = np.std(shadow_region)
                shadow_detail_visible = shadow_std > 10  # 阴影区域有足够的纹理变化
        
        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'is_night': is_night,
            'is_too_dark': is_too_dark,
            'is_overexposed': is_overexposed,
            'low_light_ratio': low_light_ratio,
            'shadow_detail_visible': shadow_detail_visible,
            'lighting_quality': 'good' if not (is_too_dark or is_overexposed) else 'poor'
        }
    
    def _detect_objects_yolo(self, image_path: Path) -> Set[str]:
        """使用YOLO检测对象"""
        if not self.use_yolo:
            return set()
        
        try:
            results = self.yolo_model(str(image_path), verbose=False)
            detected_objects = set()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:  # 置信度阈值
                            class_name = self.yolo_model.names[class_id].lower()
                            
                            # 映射到我们的场景类型
                            for scene_type, keywords in self.scene_types.items():
                                if any(keyword in class_name for keyword in keywords):
                                    detected_objects.add(scene_type)
                                    break
                            
                            # 特殊处理一些类别
                            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                                detected_objects.add('vehicle')
                            elif class_name in ['person', 'people']:
                                detected_objects.add('person_animal')
            
            return detected_objects
            
        except Exception as e:
            print(f"YOLO检测失败: {e}")
            return set()
    
    def _detect_objects_opencv(self, image: np.ndarray) -> Set[str]:
        """使用OpenCV检测对象"""
        if not self.use_cascade:
            return set()
        
        detected_objects = set()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            # 人脸检测
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                detected_objects.add('person_animal')
            
            # 车辆检测（如果可用）
            if self.car_cascade is not None:
                cars = self.car_cascade.detectMultiScale(gray, 1.1, 4)
                if len(cars) > 0:
                    detected_objects.add('vehicle')
        
        except Exception as e:
            print(f"OpenCV检测失败: {e}")
        
        return detected_objects
    
    def _detect_scene_features(self, image: np.ndarray) -> Set[str]:
        """基于图像特征检测场景类型"""
        detected_scenes = set()
        
        # 转换为HSV进行颜色分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测植被（绿色区域）
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
        
        if green_ratio > 0.2:  # 20%以上的绿色区域
            detected_scenes.add('vegetation')
        
        # 检测道路特征（灰色水平线条）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫线变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            horizontal_lines = 0
            for rho, theta in lines[:, 0]:
                # 检测接近水平的线条
                if abs(theta - np.pi/2) < np.pi/6:  # 30度范围内
                    horizontal_lines += 1
            
            if horizontal_lines > 5:
                detected_scenes.add('road')
        
        # 检测建筑特征（垂直边缘和矩形结构）
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        vertical_ratio = np.sum(vertical_lines > 0) / (image.shape[0] * image.shape[1])
        
        if vertical_ratio > 0.05:  # 5%以上的垂直线条
            detected_scenes.add('building')
        
        return detected_scenes
    
    def _detect_text_ocr(self, image_path: Path) -> bool:
        """使用OCR检测文本"""
        if not self.use_ocr:
            return False
        
        try:
            result = self.ocr.ocr(str(image_path), cls=True)
            if result and result[0]:
                for line in result[0]:
                    if len(line) > 1 and len(line[1]) > 1:
                        confidence = line[1][1]
                        if confidence > 0.7:  # 置信度阈值
                            return True
            return False
        except Exception as e:
            print(f"OCR检测失败: {e}")
            return False
    
    def _assess_overall_quality(self, image: np.ndarray, blur_score: float, 
                              lighting_info: Dict) -> Tuple[bool, str, float]:
        """综合评估图像质量 - 更宽松的标准"""
        quality_score = 0.0
        reject_reasons = []
        
        # 模糊度评分 (40%权重) - 基于改进的模糊检测
        is_blurry, blur_quality = self._detect_blur(image)
        
        if not is_blurry:
            if blur_score > 800:
                quality_score += 0.4    # 优秀质量
            elif blur_score > 500:
                quality_score += 0.35   # 良好质量
            elif blur_score > 300:
                quality_score += 0.3    # 可接受质量
            else:
                quality_score += 0.25   # 勉强可接受
        else:
            reject_reasons.append('poor_image_quality')  # 图像质量问题（模糊/噪声/过度锐化）
        
        # 光照质量评分 (40%权重) - 更宽松的标准
        if lighting_info['lighting_quality'] == 'good':
            quality_score += 0.4
        elif not lighting_info['is_too_dark'] and lighting_info['shadow_detail_visible']:
            quality_score += 0.35  # 有阴影但细节可见
        elif lighting_info['is_night'] and not lighting_info['is_too_dark']:
            quality_score += 0.3   # 夜间图像但不是太暗
        else:
            if lighting_info['is_too_dark']:
                reject_reasons.append('extremely_dark')
            if lighting_info['is_overexposed']:
                reject_reasons.append('severely_overexposed')
        
        # 图像清晰度和细节 (20%权重) - 更宽松的标准
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detail_score = np.std(gray) / 255.0  # 标准差反映细节丰富程度
        if detail_score > 0.12:  # 降低阈值
            quality_score += 0.2
        elif detail_score > 0.08:  # 降低阈值
            quality_score += 0.15
        elif detail_score > 0.05:  # 即使细节较少也给一些分数
            quality_score += 0.1
        
        # 判断是否接受 - 只有严重问题才拒绝
        is_accepted = quality_score >= self.quality_threshold and len(reject_reasons) == 0
        
        return is_accepted, ','.join(reject_reasons) if reject_reasons else 'good', quality_score
    
    def _process_single_image(self, image_path: Path) -> Dict:
        """处理单个图像"""
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': f'无法读取图像: {image_path}'}
            
            # 检测模糊
            is_blurry, blur_score = self._detect_blur(image)
            
            # 分析光照
            lighting_info = self._analyze_lighting(image)
            
            # 综合质量评估
            is_accepted, quality_issues, quality_score = self._assess_overall_quality(
                image, blur_score, lighting_info)
            
            # 如果图像被接受，进行详细分析
            scene_types = set()
            has_text = False
            
            if is_accepted:
                # 对象检测
                if self.use_yolo:
                    yolo_objects = self._detect_objects_yolo(image_path)
                    scene_types.update(yolo_objects)
                
                # OpenCV检测作为补充
                opencv_objects = self._detect_objects_opencv(image)
                scene_types.update(opencv_objects)
                
                # 场景特征检测
                scene_features = self._detect_scene_features(image)
                scene_types.update(scene_features)
                
                # 文本检测
                has_text = self._detect_text_ocr(image_path)
                
                # 如果没有检测到特定场景，标记为other
                if not scene_types:
                    scene_types.add('other')
            
            return {
                'filepath': str(image_path),
                'is_accepted': is_accepted,
                'has_text': has_text,
                'scene_types': scene_types,
                'is_night': lighting_info['is_night'],
                'quality_score': quality_score,
                'quality_issues': quality_issues,
                'blur_score': blur_score,
                'mean_brightness': lighting_info['mean_brightness'],
                'shadow_detail_visible': lighting_info['shadow_detail_visible']
            }
            
        except Exception as e:
            return {'error': f'处理图像时出错: {e}'}
    
    def process_folder(self) -> pd.DataFrame:
        """处理整个文件夹"""
        print("开始扫描图像文件...")
        image_files = self._scan_images()
        
        if not image_files:
            print("未找到图像文件")
            return pd.DataFrame()
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 创建结果DataFrame
        results = []
        
        for i, image_path in enumerate(image_files):
            print(f"处理进度: {i+1}/{len(image_files)} - {image_path.name}")
            
            result = self._process_single_image(image_path)
            
            if 'error' in result:
                print(f"  错误: {result['error']}")
                continue
            
            self.stats['total_processed'] += 1
            
            if result['is_accepted']:
                # 接受的图像
                self.stats['accepted'] += 1
                
                # 添加到结果
                scene_type_str = ','.join(sorted(result['scene_types']))
                results.append({
                    'filepath': result['filepath'],
                    'has_text': int(result['has_text']),
                    'scene_type': scene_type_str,
                    'quality_level': 'high' if result['quality_score'] > 0.8 else 'medium',
                    'is_night': int(result['is_night']),
                    'quality_score': round(result['quality_score'], 3),
                    'blur_score': round(result['blur_score'], 2),
                    'mean_brightness': round(result['mean_brightness'], 2),
                    'shadow_detail_visible': int(result['shadow_detail_visible']),
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"  ✓ 接受 - 场景: {scene_type_str}, 文本: {'是' if result['has_text'] else '否'}, "
                      f"质量: {result['quality_score']:.3f}")
            
            else:
                # 拒绝的图像 - 移动到拒绝文件夹
                self.stats['rejected'] += 1
                
                # 统计拒绝原因
                if 'motion_blur' in result['quality_issues']:
                    self.stats['blur_rejected'] += 1
                if 'too_dark' in result['quality_issues']:
                    self.stats['low_light_rejected'] += 1
                if result['quality_score'] < 0.3:
                    self.stats['poor_quality_rejected'] += 1
                
                # 移动到拒绝文件夹
                try:
                    rejected_path = self.rejected_folder / image_path.name
                    # 如果目标文件已存在，添加数字后缀
                    counter = 1
                    while rejected_path.exists():
                        name_parts = image_path.stem, counter, image_path.suffix
                        rejected_path = self.rejected_folder / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        counter += 1
                    
                    shutil.move(str(image_path), str(rejected_path))
                    print(f"  ✗ 拒绝 - 原因: {result['quality_issues']}, 已移动到: {rejected_path.name}")
                
                except Exception as e:
                    print(f"  ✗ 拒绝但移动失败: {e}")
        
        # 创建DataFrame并保存
        df = pd.DataFrame(results)
        
        if not df.empty:
            df.to_csv(self.csv_path, index=False)
            print(f"\n结果已保存到: {self.csv_path}")
        
        return df
    
    def print_statistics(self):
        """打印处理统计信息"""
        print(f"\n=== 处理统计 ===")
        print(f"总处理图像: {self.stats['total_processed']}")
        print(f"接受图像: {self.stats['accepted']}")
        print(f"拒绝图像: {self.stats['rejected']}")
        print(f"  - 运动模糊: {self.stats['blur_rejected']}")
        print(f"  - 光照不足: {self.stats['low_light_rejected']}")
        print(f"  - 质量较差: {self.stats['poor_quality_rejected']}")
        
        if self.stats['total_processed'] > 0:
            accept_rate = self.stats['accepted'] / self.stats['total_processed'] * 100
            print(f"接受率: {accept_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='自动图像筛选和标注工具')
    parser.add_argument('--root', required=True,
                       help='图像文件夹路径')
    parser.add_argument('--csv', 
                       help='CSV输出文件路径（默认在输入文件夹下创建auto_accepted_set.csv）')
    parser.add_argument('--rejected', 
                       help='拒绝图像文件夹路径（默认在输入文件夹下创建rejected_images文件夹）')
    parser.add_argument('--no-ocr', action='store_true', 
                       help='禁用OCR文本检测')
    parser.add_argument('--quality-threshold', type=float, default=0.6,
                       help='质量阈值 (0.0-1.0, 默认0.6)')
    
    args = parser.parse_args()
    
    # 检查输入路径
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"错误: 图像文件夹不存在: {args.root}")
        return
    
    # 设置输出路径
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = root_path / "auto_accepted_set.csv"
    
    if args.rejected:
        rejected_path = args.rejected
    else:
        rejected_path = None  # 将使用默认路径
    
    # 创建输出目录
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"自动图像筛选工具启动")
    print(f"输入文件夹: {root_path}")
    print(f"CSV输出: {csv_path}")
    print(f"质量阈值: {args.quality_threshold}")
    print(f"OCR检测: {'禁用' if args.no_ocr else '启用'}")
    
    # 创建筛选器并运行
    curator = AutoImageCurator(
        root_folder=str(root_path),
        csv_path=str(csv_path),
        rejected_folder=rejected_path,
        use_ocr=not args.no_ocr,
        quality_threshold=args.quality_threshold
    )
    
    # 处理图像
    df = curator.process_folder()
    
    # 显示统计信息
    curator.print_statistics()
    
    if not df.empty:
        print(f"\n=== CSV文件内容预览 ===")
        print(df.head())
        
        print(f"\n=== 场景类型统计 ===")
        scene_counts = {}
        for scene_str in df['scene_type']:
            for scene in scene_str.split(','):
                scene = scene.strip()
                if scene:
                    scene_counts[scene] = scene_counts.get(scene, 0) + 1
        
        for scene, count in sorted(scene_counts.items()):
            print(f"  {scene}: {count}")
        
        text_count = df['has_text'].sum()
        night_count = df['is_night'].sum()
        print(f"\n包含文本的图像: {text_count}/{len(df)} ({text_count/len(df)*100:.1f}%)")
        print(f"夜间图像: {night_count}/{len(df)} ({night_count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
