#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基準測試與完整性檢查 (Baseline Evaluation and Sanity Checks)
========================================================

目的：建立基準性能指標，用於後續模型比較

包含的基準方法：
1. Bicubic Upscaling（雙三次插值放大）
2. Identity/Pass-through（恆等映射）
3. OpenCV去噪（Gaussian, Bilateral, Non-local Means）
4. OpenCV銳化（Unsharp Masking, Laplacian）

評估指標：
- 有參考指標（需要 Ground Truth）：PSNR, SSIM, LPIPS
- 無參考指標：NIQE, BRISQUE, Variance of Laplacian, Tenengrad
- 無人機專用指標：Blur Extent, Luminance, Contrast, Entropy, Gradient Magnitude

輸出：
- 每個方法的排行榜 CSV
- 樣本前後對比圖
- 失敗案例分析

作者：FYP Project
日期：2025-11
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入進階指標庫
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("⚠️  警告：skimage 未安裝，將使用 OpenCV 計算 PSNR/SSIM")
    SKIMAGE_AVAILABLE = False

try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("⚠️  警告：LPIPS 未安裝，將跳過 LPIPS 計算")
    LPIPS_AVAILABLE = False

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    print("⚠️  警告：pyiqa 未安裝，將跳過 NIQE/BRISQUE 計算")
    PYIQA_AVAILABLE = False


# ==================== 配置區 ====================

class Config:
    """配置類"""
    
    # 路徑配置
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset")
    ORIGINAL_DIR = Path(r"C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP_Images")
    OUTPUT_DIR = Path(r"D:\baseline_results")
    
    # 輸出子目錄
    RESULTS_DIR = OUTPUT_DIR / "enhanced_images"
    COMPARISON_DIR = OUTPUT_DIR / "comparison_samples"
    LEADERBOARD_DIR = OUTPUT_DIR / "leaderboards"
    FAILURE_DIR = OUTPUT_DIR / "failure_cases"
    
    # 樣本數量（用於可視化）
    NUM_SAMPLES = 50  # 增加到 50 張（包含 Real-ESRGAN 對比）
    NUM_FAILURE_SAMPLES = 10
    
    # 基準方法配置
    BICUBIC_SCALE = 2  # 放大倍數（如果需要）
    
    # OpenCV 去噪參數
    GAUSSIAN_KERNEL = (5, 5)
    GAUSSIAN_SIGMA = 1.0
    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75
    NLM_H = 10
    NLM_TEMPLATE_WINDOW = 7
    NLM_SEARCH_WINDOW = 21
    
    # 銳化參數
    SHARPEN_KERNEL = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    UNSHARP_SIGMA = 1.0
    UNSHARP_AMOUNT = 1.5
    
    # 設備配置
    try:
        import torch
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        DEVICE = 'cpu'


# ==================== 基準方法 ====================

class BaselineMethods:
    """基準方法集合"""
    
    @staticmethod
    def identity(img: np.ndarray) -> np.ndarray:
        """恆等映射（不做任何處理）"""
        return img.copy()
    
    @staticmethod
    def bicubic_upscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
        """雙三次插值放大後再縮回原大小"""
        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        # 再縮回原大小（模擬超分辨率過程）
        result = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        return result
    
    @staticmethod
    def gaussian_denoise(img: np.ndarray) -> np.ndarray:
        """高斯去噪"""
        return cv2.GaussianBlur(img, Config.GAUSSIAN_KERNEL, Config.GAUSSIAN_SIGMA)
    
    @staticmethod
    def bilateral_denoise(img: np.ndarray) -> np.ndarray:
        """雙邊濾波去噪"""
        return cv2.bilateralFilter(img, Config.BILATERAL_D, 
                                   Config.BILATERAL_SIGMA_COLOR, 
                                   Config.BILATERAL_SIGMA_SPACE)
    
    @staticmethod
    def nlm_denoise(img: np.ndarray) -> np.ndarray:
        """Non-local Means 去噪"""
        return cv2.fastNlMeansDenoisingColored(img, None, 
                                               Config.NLM_H, 
                                               Config.NLM_H, 
                                               Config.NLM_TEMPLATE_WINDOW, 
                                               Config.NLM_SEARCH_WINDOW)
    
    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """銳化"""
        return cv2.filter2D(img, -1, Config.SHARPEN_KERNEL)
    
    @staticmethod
    def unsharp_mask(img: np.ndarray) -> np.ndarray:
        """Unsharp Masking 銳化"""
        blurred = cv2.GaussianBlur(img, (0, 0), Config.UNSHARP_SIGMA)
        sharpened = cv2.addWeighted(img, 1.0 + Config.UNSHARP_AMOUNT, 
                                    blurred, -Config.UNSHARP_AMOUNT, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def combined_denoise_sharpen(img: np.ndarray) -> np.ndarray:
        """組合：去噪 + 銳化"""
        denoised = BaselineMethods.bilateral_denoise(img)
        sharpened = BaselineMethods.unsharp_mask(denoised)
        return sharpened


# ==================== 評估指標 ====================

class MetricsCalculator:
    """指標計算器"""
    
    def __init__(self):
        # 初始化 LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_metric = lpips.LPIPS(net='alex').to(Config.DEVICE)
        else:
            self.lpips_metric = None
        
        # 初始化 PYIQA
        if PYIQA_AVAILABLE:
            try:
                self.niqe_metric = pyiqa.create_metric('niqe', device=Config.DEVICE)
                self.brisque_metric = pyiqa.create_metric('brisque', device=Config.DEVICE)
            except Exception as e:
                print(f"⚠️  PYIQA 初始化失敗: {e}")
                self.niqe_metric = None
                self.brisque_metric = None
        else:
            self.niqe_metric = None
            self.brisque_metric = None
    
    # ==================== 有參考指標 ====================
    
    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """計算 PSNR"""
        if SKIMAGE_AVAILABLE:
            return psnr(img1, img2, data_range=255)
        else:
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(255.0 / np.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """計算 SSIM"""
        if SKIMAGE_AVAILABLE:
            if len(img1.shape) == 3:
                return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)
            else:
                return ssim(img1, img2, data_range=255)
        else:
            # 使用 OpenCV 的簡化版本
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            
            mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
            
            sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1 ** 2
            sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2 ** 2
            sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2
            
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(np.mean(ssim_map))
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """計算 LPIPS"""
        if not LPIPS_AVAILABLE or self.lpips_metric is None:
            return None
        
        try:
            # 轉換為 torch tensor (需要 RGB 格式，範圍 [-1, 1])
            img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            
            img1_tensor = img1_tensor.to(Config.DEVICE)
            img2_tensor = img2_tensor.to(Config.DEVICE)
            
            with torch.no_grad():
                lpips_value = self.lpips_metric(img1_tensor, img2_tensor)
            
            return float(lpips_value.cpu().numpy())
        except Exception as e:
            print(f"⚠️  LPIPS 計算失敗: {e}")
            return None
    
    # ==================== 無參考指標 ====================
    
    def calculate_niqe(self, img: np.ndarray) -> Optional[float]:
        """計算 NIQE（越低越好）"""
        if self.niqe_metric is None:
            return None
        
        try:
            # 轉換為 torch tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(Config.DEVICE)
            
            with torch.no_grad():
                niqe_value = self.niqe_metric(img_tensor)
            
            return float(niqe_value.cpu().numpy())
        except Exception as e:
            print(f"⚠️  NIQE 計算失敗: {e}")
            return None
    
    def calculate_brisque(self, img: np.ndarray) -> Optional[float]:
        """計算 BRISQUE（越低越好）"""
        if self.brisque_metric is None:
            return None
        
        try:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(Config.DEVICE)
            
            with torch.no_grad():
                brisque_value = self.brisque_metric(img_tensor)
            
            return float(brisque_value.cpu().numpy())
        except Exception as e:
            print(f"⚠️  BRISQUE 計算失敗: {e}")
            return None
    
    # ==================== 銳度指標 ====================
    
    @staticmethod
    def variance_of_laplacian(img: np.ndarray) -> float:
        """拉普拉斯變異數（越高越清晰）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    @staticmethod
    def tenengrad(img: np.ndarray) -> float:
        """Tenengrad 梯度法（越高越清晰）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(gradient_magnitude))
    
    # ==================== 無人機專用指標 ====================
    
    @staticmethod
    def blur_extent_index(img: np.ndarray) -> float:
        """模糊程度指數（越低越清晰）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # 使用頻域分析
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 高頻能量比例（清晰圖像有更多高頻）
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        radius = min(center_h, center_w) // 3
        
        # 創建高頻遮罩
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > radius**2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        
        # 返回模糊指數（低頻占比）
        return 1.0 - (high_freq_energy / total_energy) if total_energy > 0 else 1.0
    
    @staticmethod
    def mean_luminance(img: np.ndarray) -> float:
        """平均亮度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(np.mean(gray))
    
    @staticmethod
    def contrast_ratio(img: np.ndarray) -> float:
        """對比度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(np.std(gray))
    
    @staticmethod
    def entropy(img: np.ndarray) -> float:
        """圖像熵（越高信息量越大）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]  # 移除零值
        return float(-np.sum(hist * np.log2(hist)))
    
    @staticmethod
    def gradient_magnitude_mean(img: np.ndarray) -> float:
        """平均梯度強度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(gradient_magnitude))
    
    # ==================== 綜合評估 ====================
    
    def evaluate_with_reference(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """有參考評估（需要 Ground Truth）"""
        metrics = {}
        
        # 基本指標
        metrics['psnr'] = self.calculate_psnr(pred, gt)
        metrics['ssim'] = self.calculate_ssim(pred, gt)
        
        # LPIPS（如果可用）
        lpips_val = self.calculate_lpips(pred, gt)
        if lpips_val is not None:
            metrics['lpips'] = lpips_val
        
        # 銳度指標（在預測圖上）
        metrics['var_laplacian'] = self.variance_of_laplacian(pred)
        metrics['tenengrad'] = self.tenengrad(pred)
        
        # 無人機專用指標
        metrics['blur_extent'] = self.blur_extent_index(pred)
        metrics['luminance'] = self.mean_luminance(pred)
        metrics['contrast'] = self.contrast_ratio(pred)
        metrics['entropy'] = self.entropy(pred)
        metrics['gradient_mean'] = self.gradient_magnitude_mean(pred)
        
        return metrics
    
    def evaluate_no_reference(self, img: np.ndarray) -> Dict[str, float]:
        """無參考評估（不需要 Ground Truth）"""
        metrics = {}
        
        # 無參考質量指標
        niqe_val = self.calculate_niqe(img)
        if niqe_val is not None:
            metrics['niqe'] = niqe_val
        
        brisque_val = self.calculate_brisque(img)
        if brisque_val is not None:
            metrics['brisque'] = brisque_val
        
        # 銳度指標
        metrics['var_laplacian'] = self.variance_of_laplacian(img)
        metrics['tenengrad'] = self.tenengrad(img)
        
        # 無人機專用指標
        metrics['blur_extent'] = self.blur_extent_index(img)
        metrics['luminance'] = self.mean_luminance(img)
        metrics['contrast'] = self.contrast_ratio(img)
        metrics['entropy'] = self.entropy(img)
        metrics['gradient_mean'] = self.gradient_magnitude_mean(img)
        
        return metrics


# ==================== 主評估器 ====================

class BaselineEvaluator:
    """基準評估器"""
    
    def __init__(self):
        self.config = Config()
        self.methods = BaselineMethods()
        self.calculator = MetricsCalculator()
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 基準方法字典
        self.baseline_methods = {
            'identity': self.methods.identity,
            'bicubic': self.methods.bicubic_upscale,
            'gaussian': self.methods.gaussian_denoise,
            'bilateral': self.methods.bilateral_denoise,
            'nlm': self.methods.nlm_denoise,
            'sharpen': self.methods.sharpen,
            'unsharp': self.methods.unsharp_mask,
            'combined': self.methods.combined_denoise_sharpen,
        }
    
    def find_degraded_images(self) -> List[Tuple[Path, str, str]]:
        """
        查找所有退化圖像
        返回：[(退化圖像路徑, 退化類型, 原始檔名), ...]
        """
        degraded_images = []
        
        if not self.config.DEGRADED_DIR.exists():
            print(f"❌ 退化圖像目錄不存在: {self.config.DEGRADED_DIR}")
            return degraded_images
        
        # 遍歷退化類型
        degradation_types = ['drone_motion_blur', 'object_motion_blur', 'low_light']
        
        for deg_type in degradation_types:
            deg_dir = self.config.DEGRADED_DIR / deg_type
            if not deg_dir.exists():
                continue
            
            # 查找所有圖像
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                for img_path in deg_dir.glob(f'*{ext}'):
                    degraded_images.append((img_path, deg_type, img_path.name))
        
        print(f"✓ 找到 {len(degraded_images)} 張退化圖像")
        return degraded_images
    
    def process_image_with_baselines(self, img_path: Path, 
                                     degradation_type: str) -> Dict[str, np.ndarray]:
        """使用所有基準方法處理單張圖像"""
        # 讀取退化圖像
        degraded = cv2.imread(str(img_path))
        if degraded is None:
            return {}
        
        results = {'degraded': degraded}
        
        # 應用所有基準方法
        for method_name, method_func in self.baseline_methods.items():
            try:
                enhanced = method_func(degraded)
                results[method_name] = enhanced
            except Exception as e:
                print(f"  ⚠️  {method_name} 處理失敗: {e}")
                continue
        
        return results
    
    def evaluate_all_baselines(self):
        """評估所有基準方法"""
        print("\n" + "=" * 80)
        print("🎯 開始基準測試評估")
        print("=" * 80)
        
        # 查找所有退化圖像
        degraded_images = self.find_degraded_images()
        if not degraded_images:
            print("❌ 沒有找到退化圖像")
            return
        
        # 初始化結果存儲
        all_results = []
        
        # 處理每張圖像
        print(f"\n📊 處理 {len(degraded_images)} 張圖像...")
        for img_path, deg_type, orig_name in tqdm(degraded_images, desc="評估進度"):
            # 處理圖像
            results = self.process_image_with_baselines(img_path, deg_type)
            if not results:
                continue
            
            degraded = results['degraded']
            
            # 評估每個方法（無參考指標）
            for method_name in self.baseline_methods.keys():
                if method_name not in results:
                    continue
                
                enhanced = results[method_name]
                
                # 計算指標
                metrics = self.calculator.evaluate_no_reference(enhanced)
                
                # 記錄結果
                result_entry = {
                    'image_name': orig_name,
                    'degradation_type': deg_type,
                    'method': method_name,
                    **metrics
                }
                all_results.append(result_entry)
        
        # 保存結果到 DataFrame
        df = pd.DataFrame(all_results)
        
        # 生成排行榜
        self.generate_leaderboards(df)
        
        # 保存完整結果
        full_results_path = self.config.LEADERBOARD_DIR / "full_results.csv"
        df.to_csv(full_results_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 完整結果已保存: {full_results_path}")
        
        # 生成可視化樣本
        self.generate_comparison_samples(degraded_images[:Config.NUM_SAMPLES])
        
        # 識別失敗案例
        self.identify_failure_cases(df, degraded_images)
        
        print("\n" + "=" * 80)
        print("✅ 基準測試評估完成！")
        print("=" * 80)
        print(f"\n📁 所有結果已保存至: {self.config.OUTPUT_DIR}")
    
    def generate_leaderboards(self, df: pd.DataFrame):
        """生成排行榜"""
        print("\n📊 生成排行榜...")
        
        # 按退化類型分組
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            # 計算平均指標
            leaderboard = df_type.groupby('method').agg({
                'var_laplacian': 'mean',
                'tenengrad': 'mean',
                'blur_extent': 'mean',
                'luminance': 'mean',
                'contrast': 'mean',
                'entropy': 'mean',
                'gradient_mean': 'mean',
            }).round(4)
            
            # 添加 NIQE/BRISQUE（如果有）
            if 'niqe' in df_type.columns:
                leaderboard['niqe'] = df_type.groupby('method')['niqe'].mean()
            if 'brisque' in df_type.columns:
                leaderboard['brisque'] = df_type.groupby('method')['brisque'].mean()
            
            # 計算排名（多個指標的綜合排名）
            # 銳度指標：越高越好
            leaderboard['sharpness_score'] = (
                leaderboard['var_laplacian'] / leaderboard['var_laplacian'].max() +
                leaderboard['tenengrad'] / leaderboard['tenengrad'].max()
            ) / 2
            
            # 模糊指標：越低越好
            leaderboard['blur_score'] = 1 - (leaderboard['blur_extent'] / leaderboard['blur_extent'].max())
            
            # 綜合得分
            leaderboard['overall_score'] = (leaderboard['sharpness_score'] + leaderboard['blur_score']) / 2
            
            # 排序
            leaderboard = leaderboard.sort_values('overall_score', ascending=False)
            
            # 保存
            output_path = self.config.LEADERBOARD_DIR / f"leaderboard_{deg_type}.csv"
            leaderboard.to_csv(output_path, encoding='utf-8-sig')
            print(f"  ✓ {deg_type} 排行榜已保存: {output_path.name}")
            
            # 打印前3名
            print(f"\n  🏆 {deg_type} Top 3:")
            for i, (method, row) in enumerate(leaderboard.head(3).iterrows(), 1):
                print(f"    {i}. {method:12s} (得分: {row['overall_score']:.4f})")
    
    def generate_comparison_samples(self, sample_images: List[Tuple[Path, str, str]]):
        """生成對比樣本"""
        print(f"\n🖼️  生成對比樣本（前 {len(sample_images)} 張）...")
        
        for img_path, deg_type, orig_name in tqdm(sample_images, desc="生成樣本"):
            # 處理圖像
            results = self.process_image_with_baselines(img_path, deg_type)
            if not results:
                continue
            
            # 創建對比圖
            num_methods = len(results)
            cols = 4
            rows = (num_methods + cols - 1) // cols
            
            fig_width = cols * 4
            fig_height = rows * 3
            
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            axes = axes.flatten() if num_methods > 1 else [axes]
            
            for idx, (method_name, img) in enumerate(results.items()):
                if idx >= len(axes):
                    break
                
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(method_name, fontsize=10)
                axes[idx].axis('off')
            
            # 隱藏多餘的子圖
            for idx in range(num_methods, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # 保存
            output_name = f"{Path(orig_name).stem}_{deg_type}_comparison.png"
            output_path = self.config.COMPARISON_DIR / output_name
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 對比樣本已保存至: {self.config.COMPARISON_DIR}")
    
    def identify_failure_cases(self, df: pd.DataFrame, 
                               all_images: List[Tuple[Path, str, str]]):
        """識別失敗案例（性能最差的樣本）"""
        print(f"\n🔍 識別失敗案例（前 {Config.NUM_FAILURE_SAMPLES} 個）...")
        
        # 對於每個方法，找出最差的樣本
        for method_name in self.baseline_methods.keys():
            df_method = df[df['method'] == method_name]
            
            # 按 blur_extent 排序（越高越模糊）
            worst_cases = df_method.nsmallest(Config.NUM_FAILURE_SAMPLES, 'var_laplacian')
            
            # 保存失敗案例列表
            failure_list_path = self.config.FAILURE_DIR / f"failure_cases_{method_name}.csv"
            worst_cases.to_csv(failure_list_path, index=False, encoding='utf-8-sig')
            
            print(f"  ✓ {method_name} 失敗案例已保存")


# ==================== 主程序 ====================

def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("🚀 基準測試與完整性檢查")
    print("=" * 80)
    print("\n📋 配置資訊:")
    print(f"  退化圖像目錄: {Config.DEGRADED_DIR}")
    print(f"  輸出目錄: {Config.OUTPUT_DIR}")
    print(f"  設備: {Config.DEVICE}")
    print(f"  LPIPS 可用: {'是' if LPIPS_AVAILABLE else '否'}")
    print(f"  PYIQA 可用: {'是' if PYIQA_AVAILABLE else '否'}")
    
    # 創建評估器
    evaluator = BaselineEvaluator()
    
    # 執行評估
    evaluator.evaluate_all_baselines()
    
    print("\n✅ 所有任務完成！")
    print(f"\n📁 請檢查輸出目錄: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

