"""
影像降質合成腳本 (Image Degradation Synthesis Script)
Phase 2 - 生成訓練用的降質影像

功能 (Features):
1. 全局運動模糊 (Global Motion Blur)
2. 物體運動模糊 (Object Motion Blur) - 使用 YOLOv8-seg 識別車輛等物體
3. 低光照降質 (Low-light Degradation)

作者: AI Assistant
日期: 2025-11-04
"""

import os
import json
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import argparse
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 嘗試導入 ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics 未安裝，物體運動模糊功能將被停用")
    print("安裝指令: pip install ultralytics")


class MotionBlurSynthesizer:
    """全局運動模糊合成器 (Global Motion Blur Synthesizer)"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"MotionBlurSynthesizer 使用設備: {device}")
    
    def generate_motion_kernel(self, kernel_size: int, angle: float, 
                              motion_length: float) -> np.ndarray:
        """
        生成運動模糊核 (Generate motion blur kernel / PSF)
        
        Args:
            kernel_size: 核大小 (kernel size)
            angle: 運動角度，度數 (motion angle in degrees)
            motion_length: 運動長度 (motion length in pixels)
        
        Returns:
            運動模糊核 (motion blur kernel)
        """
        # 創建空白核
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # 將角度轉換為弧度
        angle_rad = np.deg2rad(angle)
        
        # 計算運動軌跡
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 在核上畫出運動軌跡
        for i in range(int(motion_length)):
            offset = i - motion_length / 2
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # 正規化
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        
        return kernel
    
    def apply_motion_blur_fft(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        使用 FFT 應用運動模糊 (Apply motion blur using FFT)
        
        Args:
            image: 輸入影像 BGR (input image)
            kernel: 運動模糊核 (motion blur kernel)
        
        Returns:
            模糊後的影像 (blurred image)
        """
        # 將影像轉換為浮點數
        img_float = image.astype(np.float32) / 255.0
        
        # 對每個通道分別處理
        result = np.zeros_like(img_float)
        
        for i in range(3):  # BGR channels
            channel = img_float[:, :, i]
            
            # 使用 OpenCV 的 filter2D（內部使用 FFT 優化）
            blurred = cv2.filter2D(channel, -1, kernel)
            result[:, :, i] = blurred
        
        # 轉換回 uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    def apply_rolling_shutter(self, image: np.ndarray, 
                             shear_amount: float = 0.01) -> np.ndarray:
        """
        應用捲簾快門效果 (Apply rolling shutter effect)
        
        Args:
            image: 輸入影像 (input image)
            shear_amount: 扭曲量 (shear amount)
        
        Returns:
            扭曲後的影像 (warped image)
        """
        h, w = image.shape[:2]
        
        # 創建仿射變換矩陣（行方向的剪切）
        shear_matrix = np.array([
            [1, shear_amount, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        # 應用變換
        warped = cv2.warpAffine(image, shear_matrix, (w, h), 
                               borderMode=cv2.BORDER_REFLECT)
        
        return warped
    
    def synthesize(self, image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
        """
        合成全局運動模糊 (Synthesize global motion blur)
        
        Args:
            image: 輸入影像 (input image)
            params: 參數字典 (parameters dictionary)
                - kernel_size: 核大小 (default: 15)
                - angle: 角度 0-360 (default: random)
                - motion_length: 運動長度 (default: random 5-25)
                - use_rolling_shutter: 是否使用捲簾快門 (default: False)
                - shear_amount: 扭曲量 (default: 0.01)
        
        Returns:
            (降質影像, 實際參數)
        """
        # 設定預設參數
        kernel_size = params.get('kernel_size', 15)
        angle = params.get('angle', random.uniform(0, 360))
        motion_length = params.get('motion_length', random.uniform(5, 25))
        use_rs = params.get('use_rolling_shutter', False)
        shear_amount = params.get('shear_amount', 0.01)
        
        # 複製影像
        degraded = image.copy()
        
        # 可選：應用捲簾快門
        if use_rs:
            degraded = self.apply_rolling_shutter(degraded, shear_amount)
        
        # 生成運動模糊核
        kernel = self.generate_motion_kernel(kernel_size, angle, motion_length)
        
        # 應用運動模糊
        degraded = self.apply_motion_blur_fft(degraded, kernel)
        
        # 記錄實際參數
        actual_params = {
            'mode': 'global_motion_blur',
            'kernel_size': kernel_size,
            'angle': float(angle),
            'motion_length': float(motion_length),
            'use_rolling_shutter': use_rs,
            'shear_amount': float(shear_amount) if use_rs else 0.0
        }
        
        return degraded, actual_params


class ObjectMotionBlurSynthesizer:
    """物體運動模糊合成器 (Object Motion Blur Synthesizer)"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                
                print("載入 YOLOv8-seg 模型...")
                self.model = YOLO('yolov8n-seg.pt')  
                self.model.to(device)
                print(f"ObjectMotionBlurSynthesizer 使用設備: {device}")
            except Exception as e:
                print(f"載入 YOLOv8 模型失敗: {e}")
                self.model = None
        
        
        self.vehicle_classes = {
            2: 'car',           
            3: 'motorcycle',    
            5: 'bus',           
            7: 'truck',         
            1: 'bicycle',       
        }
    
    def detect_objects(self, image: np.ndarray, 
                      conf_threshold: float = 0.2) -> List[Dict]:
        """
        偵測影像中的物體 (Detect objects in image)
        
        Args:
            image: 輸入影像 BGR (input image)
            conf_threshold: 信心度閾值 (confidence threshold)
        
        Returns:
            偵測結果列表 (list of detections)
        """
        if self.model is None:
            return []
        
        # 執行推論
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        
        # 解析結果
        for result in results:
            if result.masks is None:
                continue
            
            masks = result.masks.data.cpu().numpy()  # 遮罩
            boxes = result.boxes.data.cpu().numpy()   # 邊界框
            
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                class_id = int(box[5])
                confidence = float(box[4])
                
                # 只保留車輛類別
                if class_id in self.vehicle_classes:
                    # 調整遮罩大小到原始影像尺寸
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.vehicle_classes[class_id],
                        'confidence': confidence,
                        'mask': mask_binary,
                        'bbox': box[:4]  # x1, y1, x2, y2
                    })
        
        return detections
    
    def feather_mask(self, mask: np.ndarray, feather_amount: int = 5) -> np.ndarray:
        """
        羽化遮罩邊緣 (Feather mask edges)
        
        Args:
            mask: 二值遮罩 (binary mask)
            feather_amount: 羽化量（像素）(feather amount in pixels)
        
        Returns:
            羽化後的遮罩 (feathered mask)
        """
        # 高斯模糊實現羽化
        mask_float = mask.astype(np.float32)
        feathered = cv2.GaussianBlur(mask_float, (feather_amount*2+1, feather_amount*2+1), 0)
        
        return feathered
    
    def apply_directional_blur(self, image: np.ndarray, angle: float, 
                              motion_length: int) -> np.ndarray:
        """
        應用方向性模糊 (Apply directional blur)
        
        Args:
            image: 輸入影像 (input image)
            angle: 運動角度 (motion angle)
            motion_length: 運動長度 (motion length)
        
        Returns:
            模糊後的影像 (blurred image)
        """
        # 生成運動模糊核
        kernel_size = motion_length * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        for i in range(motion_length):
            offset = i - motion_length / 2
            x = int(center + offset * cos_a)
            y = int(center + offset * sin_a)
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        
        # 應用模糊
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def synthesize(self, image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
        """
        合成物體運動模糊 (Synthesize object motion blur)
        
        Args:
            image: 輸入影像 (input image)
            params: 參數字典 (parameters dictionary)
                - conf_threshold: YOLO 信心度閾值 (default: 0.2)
                - angle: 運動角度 (default: random)
                - motion_length: 運動長度 (default: random 5-15)
                - feather_amount: 羽化量 (default: 5)
        
        Returns:
            (降質影像, 實際參數)
        """
        if self.model is None:
            print("YOLOv8 模型未載入，返回原始影像")
            return image.copy(), {'mode': 'object_motion_blur', 'error': 'model_not_loaded'}
        
        # 設定參數
        conf_threshold = params.get('conf_threshold', 0.2)
        angle = params.get('angle', random.uniform(0, 360))
        motion_length = params.get('motion_length', random.randint(5, 15))
        feather_amount = params.get('feather_amount', 5)
        
        # 偵測物體
        detections = self.detect_objects(image, conf_threshold)
        
        if len(detections) == 0:
            # 沒有偵測到物體，返回原始影像
            return image.copy(), {
                'mode': 'object_motion_blur',
                'objects_detected': 0,
                'note': 'no_objects_detected'
            }
        
        # 開始合成
        result = image.copy()
        
        for det in detections:
            mask = det['mask']
            
            # 羽化遮罩
            mask_feathered = self.feather_mask(mask, feather_amount)
            mask_3ch = np.stack([mask_feathered] * 3, axis=-1)
            
            # 對物體區域應用運動模糊
            blurred_obj = self.apply_directional_blur(image, angle, motion_length)
            
            # 混合模糊物體和原始背景
            result = (mask_3ch * blurred_obj + (1 - mask_3ch) * result).astype(np.uint8)
        
        # 記錄實際參數
        actual_params = {
            'mode': 'object_motion_blur',
            'objects_detected': len(detections),
            'angle': float(angle),
            'motion_length': int(motion_length),
            'feather_amount': int(feather_amount),
            'conf_threshold': float(conf_threshold),
            'detected_classes': [det['class_name'] for det in detections]
        }
        
        return result, actual_params


class LowLightSynthesizer:
    """低光照降質合成器 (Low-light Degradation Synthesizer)"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"LowLightSynthesizer 使用設備: {device}")
    
    def apply_exposure_reduction(self, image: np.ndarray, 
                                 exposure_factor: float) -> np.ndarray:
        """
        降低曝光 (Reduce exposure)
        
        Args:
            image: 輸入影像 (input image)
            exposure_factor: 曝光因子 0-1 (exposure factor)
        
        Returns:
            降低曝光後的影像 (exposure-reduced image)
        """
        img_float = image.astype(np.float32) / 255.0
        reduced = img_float * exposure_factor
        return (reduced * 255).astype(np.uint8)
    
    def apply_white_balance_shift(self, image: np.ndarray, 
                                  shift_factors: Tuple[float, float, float]) -> np.ndarray:
        """
        應用白平衡偏移 (Apply white balance shift)
        
        Args:
            image: 輸入影像 BGR (input image)
            shift_factors: (B, G, R) 通道因子 (channel factors)
        
        Returns:
            偏移後的影像 (shifted image)
        """
        img_float = image.astype(np.float32)
        
        # 對每個通道分別乘以因子
        for i in range(3):
            img_float[:, :, i] *= shift_factors[i]
        
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    def apply_tone_curve(self, image: np.ndarray, 
                        black_level: float = 0.1,
                        highlight_rolloff: float = 0.9) -> np.ndarray:
        """
        應用色調曲線 (Apply tone curve)
        
        Args:
            image: 輸入影像 (input image)
            black_level: 黑色提升 (black level lift)
            highlight_rolloff: 高光壓縮 (highlight roll-off)
        
        Returns:
            調整後的影像 (tone-adjusted image)
        """
        img_float = image.astype(np.float32) / 255.0
        
        # 應用 S 曲線調整
        # 提升黑色（陰影區域變亮）
        img_float = img_float * (1 - black_level) + black_level
        
        # 壓縮高光（亮區變暗）
        img_float = np.where(img_float > highlight_rolloff,
                            highlight_rolloff + (img_float - highlight_rolloff) * 0.5,
                            img_float)
        
        return (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    
    def add_noise(self, image: np.ndarray, 
                  shot_noise_scale: float = 0.01,
                  read_noise_std: float = 5.0) -> np.ndarray:
        """
        添加噪聲 (Add noise)
        
        Args:
            image: 輸入影像 (input image)
            shot_noise_scale: 散粒噪聲尺度 (shot noise scale)
            read_noise_std: 讀取噪聲標準差 (read noise std)
        
        Returns:
            含噪聲的影像 (noisy image)
        """
        img_float = image.astype(np.float32)
        
        # 散粒噪聲（與亮度相關）Shot noise
        shot_noise = np.random.randn(*image.shape) * np.sqrt(img_float * shot_noise_scale)
        
        # 讀取噪聲（常數）Read noise
        read_noise = np.random.randn(*image.shape) * read_noise_std
        
        # 添加噪聲
        noisy = img_float + shot_noise + read_noise
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_jpeg_compression(self, image: np.ndarray, 
                              quality: int = 70) -> np.ndarray:
        """
        應用 JPEG 壓縮 (Apply JPEG compression)
        
        Args:
            image: 輸入影像 (input image)
            quality: JPEG 品質 0-100 (JPEG quality)
        
        Returns:
            壓縮後的影像 (compressed image)
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return decoded
    
    def synthesize(self, image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
        """
        合成低光照降質 (Synthesize low-light degradation)
        
        Args:
            image: 輸入影像 (input image)
            params: 參數字典 (parameters dictionary)
                - exposure_factor: 曝光因子 (default: random 0.2-0.5)
                - wb_shift: 白平衡偏移 (default: random)
                - black_level: 黑色提升 (default: 0.05)
                - highlight_rolloff: 高光壓縮 (default: 0.9)
                - shot_noise_scale: 散粒噪聲 (default: 0.01)
                - read_noise_std: 讀取噪聲 (default: random 3-8)
                - jpeg_quality: JPEG 品質 (default: random 60-85)
                - use_jpeg: 是否使用 JPEG 壓縮 (default: True)
        
        Returns:
            (降質影像, 實際參數)
        """
        # 設定參數
        exposure_factor = params.get('exposure_factor', random.uniform(0.2, 0.5))
        wb_shift = params.get('wb_shift', (
            random.uniform(0.9, 1.1),   # B
            random.uniform(0.95, 1.05), # G
            random.uniform(0.85, 0.95)  # R (偏冷色調)
        ))
        black_level = params.get('black_level', 0.05)
        highlight_rolloff = params.get('highlight_rolloff', 0.9)
        shot_noise_scale = params.get('shot_noise_scale', 0.01)
        read_noise_std = params.get('read_noise_std', random.uniform(3, 8))
        jpeg_quality = params.get('jpeg_quality', random.randint(60, 85))
        use_jpeg = params.get('use_jpeg', True)
        
        # 步驟 1: 降低曝光
        degraded = self.apply_exposure_reduction(image, exposure_factor)
        
        # 步驟 2: 白平衡偏移
        degraded = self.apply_white_balance_shift(degraded, wb_shift)
        
        # 步驟 3: 色調曲線調整
        degraded = self.apply_tone_curve(degraded, black_level, highlight_rolloff)
        
        # 步驟 4: 添加噪聲
        degraded = self.add_noise(degraded, shot_noise_scale, read_noise_std)
        
        # 步驟 5: 可選 JPEG 壓縮
        if use_jpeg:
            degraded = self.apply_jpeg_compression(degraded, jpeg_quality)
        
        # 記錄實際參數
        actual_params = {
            'mode': 'low_light',
            'exposure_factor': float(exposure_factor),
            'wb_shift_b': float(wb_shift[0]),
            'wb_shift_g': float(wb_shift[1]),
            'wb_shift_r': float(wb_shift[2]),
            'black_level': float(black_level),
            'highlight_rolloff': float(highlight_rolloff),
            'shot_noise_scale': float(shot_noise_scale),
            'read_noise_std': float(read_noise_std),
            'jpeg_quality': int(jpeg_quality) if use_jpeg else None,
            'use_jpeg': use_jpeg
        }
        
        return degraded, actual_params


def read_image_with_timeout(image_path: str, timeout: float = 5.0) -> Optional[np.ndarray]:
    """
    使用超時機制讀取影像（避免 OneDrive 下載卡住）
    
    Args:
        image_path: 影像路徑
        timeout: 超時時間（秒）
    
    Returns:
        影像陣列或 None
    """
    def _read_image():
        return cv2.imread(image_path)
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_read_image)
            image = future.result(timeout=timeout)
            return image
    except TimeoutError:
        return None
    except Exception:
        return None


class DegradationPipeline:
    """降質處理流程 (Degradation Processing Pipeline)"""
    
    def __init__(self, output_dir: str, device: str = 'auto', read_timeout: float = 10.0):
        """
        初始化降質流程
        
        Args:
            output_dir: 輸出目錄 (output directory)
            device: 計算設備 'cuda', 'cpu', 或 'auto' (computing device)
            read_timeout: 影像讀取超時時間（秒）
        """
        # 設定設備
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.read_timeout = read_timeout
        
        print(f"\n=== 降質合成流程初始化 ===")
        print(f"使用設備: {self.device}")
        print(f"影像讀取超時: {read_timeout} 秒")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
        
        # 創建輸出目錄
        self.output_dir = Path(output_dir)
        self.degraded_dir = self.output_dir / 'degraded'
        self.metadata_dir = self.output_dir / 'metadata'
        
        self.degraded_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各種降質合成器
        print("\n初始化降質合成器...")
        self.global_blur = MotionBlurSynthesizer(device=self.device)
        self.object_blur = ObjectMotionBlurSynthesizer(device=self.device)
        self.low_light = LowLightSynthesizer(device=self.device)
        
        # 用於記錄配對資訊
        self.pairs = []
        
        # 統計資訊
        self.stats = {
            'processed': 0,
            'skipped_timeout': 0,
            'skipped_error': 0,
            'total_degraded': 0
        }
    
    def process_single_image(self, image_path: str, split: str = 'train',
                           enable_global_blur: bool = True,
                           enable_object_blur: bool = True,
                           enable_low_light: bool = True) -> List[Dict]:
        """
        處理單張影像，生成多種降質版本
        
        Args:
            image_path: 影像路徑 (image path)
            split: 資料集分割 'train'/'val'/'test'
            enable_global_blur: 啟用全局模糊 (enable global blur)
            enable_object_blur: 啟用物體模糊 (enable object blur)
            enable_low_light: 啟用低光照 (enable low-light)
        
        Returns:
            生成的降質影像資訊列表 (list of degraded image info)
        """
        # 讀取影像（使用超時機制）
        image = read_image_with_timeout(image_path, timeout=self.read_timeout)
        if image is None:
            # 檢查是否超時
            if not os.path.exists(image_path):
                self.stats['skipped_error'] += 1
                # 靜默跳過，不輸出錯誤訊息
            else:
                self.stats['skipped_timeout'] += 1
                # 靜默跳過，不輸出錯誤訊息
            return []
        
        self.stats['processed'] += 1
        
        # 獲取影像基本資訊
        image_name = Path(image_path).stem
        results = []
        
        # 1. 全局運動模糊
        if enable_global_blur:
            try:
                degraded, params = self.global_blur.synthesize(image, {})
                
                # 儲存降質影像
                degraded_filename = f"{image_name}_global_blur.jpg"
                degraded_path = self.degraded_dir / degraded_filename
                cv2.imwrite(str(degraded_path), degraded)
                
                # 儲存元數據
                metadata_filename = f"{image_name}_global_blur.json"
                metadata_path = self.metadata_dir / metadata_filename
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)
                
                results.append({
                    'degraded_path': str(degraded_path),
                    'target_path': image_path,
                    'split': split,
                    'mode': 'global_blur',
                    'metadata_path': str(metadata_path)
                })
                self.stats['total_degraded'] += 1
            except Exception as e:
                pass  # 靜默跳過錯誤
        
        # 2. 物體運動模糊
        if enable_object_blur and YOLO_AVAILABLE:
            try:
                degraded, params = self.object_blur.synthesize(image, {})
                
                # 只有當偵測到物體時才儲存
                if params.get('objects_detected', 0) > 0:
                    degraded_filename = f"{image_name}_object_blur.jpg"
                    degraded_path = self.degraded_dir / degraded_filename
                    cv2.imwrite(str(degraded_path), degraded)
                    
                    metadata_filename = f"{image_name}_object_blur.json"
                    metadata_path = self.metadata_dir / metadata_filename
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(params, f, indent=2, ensure_ascii=False)
                    
                    results.append({
                        'degraded_path': str(degraded_path),
                        'target_path': image_path,
                        'split': split,
                        'mode': 'object_blur',
                        'metadata_path': str(metadata_path)
                    })
                    self.stats['total_degraded'] += 1
            except Exception as e:
                pass  # 靜默跳過錯誤
        
        # 3. 低光照降質
        if enable_low_light:
            try:
                degraded, params = self.low_light.synthesize(image, {})
                
                degraded_filename = f"{image_name}_low_light.jpg"
                degraded_path = self.degraded_dir / degraded_filename
                cv2.imwrite(str(degraded_path), degraded)
                
                metadata_filename = f"{image_name}_low_light.json"
                metadata_path = self.metadata_dir / metadata_filename
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)
                
                results.append({
                    'degraded_path': str(degraded_path),
                    'target_path': image_path,
                    'split': split,
                    'mode': 'low_light',
                    'metadata_path': str(metadata_path)
                })
                self.stats['total_degraded'] += 1
            except Exception as e:
                pass  # 靜默跳過錯誤
        
        return results
    
    def process_dataset(self, split_dir: str,
                       enable_global_blur: bool = True,
                       enable_object_blur: bool = True,
                       enable_low_light: bool = True,
                       max_images: Optional[int] = None):
        """
        處理整個資料集
        
        Args:
            split_dir: 資料分割目錄，包含 train_list.txt, val_list.txt, test_list.txt
            enable_global_blur: 啟用全局模糊
            enable_object_blur: 啟用物體模糊
            enable_low_light: 啟用低光照
            max_images: 最大處理影像數（用於測試）
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
            print(f"\n=== 處理 {split_name} 集 ===")
            
            # 限制處理數量（測試用）
            if max_images:
                image_paths = image_paths[:max_images]
            
            # 使用進度條處理
            for image_path in tqdm(image_paths, desc=f"處理 {split_name}"):
                results = self.process_single_image(
                    image_path, 
                    split=split_name,
                    enable_global_blur=enable_global_blur,
                    enable_object_blur=enable_object_blur,
                    enable_low_light=enable_low_light
                )
                
                self.pairs.extend(results)
        
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
            'device': self.device,
            'read_timeout': self.read_timeout,
            'total_pairs': len(self.pairs),
            'by_split': {},
            'by_mode': {},
            'statistics': self.stats
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
        print(f"成功處理影像: {self.stats['processed']}")
        print(f"跳過（超時）: {self.stats['skipped_timeout']}")
        print(f"跳過（檔案不存在）: {self.stats['skipped_error']}")
        print(f"總降質影像對數: {summary['total_pairs']}")
        print(f"按分割統計: {summary['by_split']}")
        print(f"按模式統計: {summary['by_mode']}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='影像降質合成工具 (Image Degradation Synthesis Tool)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例 (Usage Examples):

1. 處理完整資料集 (Process full dataset):
   python degradation_synthesis.py --split_dir ./data_split_results --output_dir ./degraded_data

2. 僅處理訓練集的前 10 張影像進行測試 (Test with 10 images):
   python degradation_synthesis.py --split_dir ./data_split_results --output_dir ./test_output --max_images 10

3. 只生成全局模糊和低光照，不生成物體模糊 (Only global blur and low-light):
   python degradation_synthesis.py --split_dir ./data_split_results --output_dir ./degraded_data --no_object_blur

4. 使用 CPU 處理 (Force CPU):
   python degradation_synthesis.py --split_dir ./data_split_results --output_dir ./degraded_data --device cpu
        """
    )
    
    parser.add_argument('--split_dir', type=str, required=True,
                       help='資料分割目錄路徑，包含 train_list.txt 等檔案')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='輸出目錄路徑')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='計算設備: auto(自動), cuda(GPU), cpu')
    parser.add_argument('--max_images', type=int, default=None,
                       help='每個分割最大處理影像數（用於測試）')
    parser.add_argument('--no_global_blur', action='store_true',
                       help='停用全局運動模糊')
    parser.add_argument('--no_object_blur', action='store_true',
                       help='停用物體運動模糊')
    parser.add_argument('--no_low_light', action='store_true',
                       help='停用低光照降質')
    parser.add_argument('--read_timeout', type=float, default=10.0,
                       help='影像讀取超時時間（秒），用於跳過 OneDrive 未下載的檔案')
    
    args = parser.parse_args()
    
    # 創建降質流程
    pipeline = DegradationPipeline(
        output_dir=args.output_dir,
        device=args.device,
        read_timeout=args.read_timeout
    )
    
    # 處理資料集
    pipeline.process_dataset(
        split_dir=args.split_dir,
        enable_global_blur=not args.no_global_blur,
        enable_object_blur=not args.no_object_blur,
        enable_low_light=not args.no_low_light,
        max_images=args.max_images
    )
    
    print("\n✓ 降質合成完成！")


if __name__ == '__main__':
    main()

