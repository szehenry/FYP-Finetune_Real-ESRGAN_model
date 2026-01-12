#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 基準評估（單獨運行版本 v2 - 含 Checkpoint 恢復）
=================================================================

改進點：
1. Checkpoint 恢復機制（隨時暫停/繼續）
2. 優化內存管理（TILE=512, FP32）
3. 更強的錯誤處理（跳過失敗圖像）
4. 與 baseline_evaluation_with_gt-HenryCC.py 一致的配置

作者：FYP Project
日期：2025-12
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import random
import gc
import signal
from contextlib import contextmanager

# 設置 UTF-8 輸出（修復 Windows 編碼問題）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 設置隨機種子
random.seed(42)
np.random.seed(42)

# 導入 Real-ESRGAN
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    print("✓ Real-ESRGAN 導入成功")
except ImportError as e:
    print(f"❌ 無法導入 Real-ESRGAN: {e}")
    print("請確保已安裝: pip install realesrgan basicsr")
    sys.exit(1)

# 導入基準方法
try:
    from baseline_evaluation import MetricsCalculator, Config as BaseConfig
    print("✓ 成功導入基準方法模組")
except ImportError as e:
    print(f"❌ 無法導入 baseline_evaluation 模組: {e}")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==================== 超時處理 ====================

class TimeoutException(Exception):
    """超時異常"""
    pass

@contextmanager
def time_limit(seconds):
    """
    超時上下文管理器（Windows 兼容版本）
    
    注意：在 Windows 上，signal.alarm 不可用
    這裡使用簡單的時間檢查機制
    """
    import time
    import threading
    
    class TimerClass:
        def __init__(self):
            self.timed_out = False
            self.timer = None
        
        def timeout_handler(self):
            self.timed_out = True
        
        def start(self, seconds):
            self.timer = threading.Timer(seconds, self.timeout_handler)
            self.timer.daemon = True
            self.timer.start()
        
        def cancel(self):
            if self.timer:
                self.timer.cancel()
    
    timer_obj = TimerClass()
    timer_obj.start(seconds)
    
    try:
        yield timer_obj
    finally:
        # ✅ 修復：先取消計時器，再檢查是否超時
        was_timed_out = timer_obj.timed_out
        timer_obj.cancel()
        if was_timed_out:
            raise TimeoutException(f"操作超時（>{seconds}秒）")


# ==================== 配置區 ====================

class RealESRGANOnlyConfig(BaseConfig):
    """Real-ESRGAN 單獨評估配置"""
    
    # 路徑配置
    PAIRS_CSV = Path(r"D:\degraded_full_dataset\pairs.csv")
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset\degraded")
    ORIGINAL_DIR = Path(r"D:\FYP_Images")
    
    # Real-ESRGAN 配置
    REALESRGAN_ROOT = Path(r"D:\Real-ESRGAN")
    MODEL_PATH = REALESRGAN_ROOT / "experiments" / "pretrained_models" / "RealESRGAN_x4plus.pth"
    
    # 輸出配置
    OUTPUT_DIR = Path(r"D:\baseline_results_realesrgan_only")
    RESULTS_DIR = OUTPUT_DIR / "results"
    COMPARISON_DIR = OUTPUT_DIR / "comparison_samples"
    LEADERBOARD_DIR = OUTPUT_DIR / "leaderboards"
    FAILURE_DIR = OUTPUT_DIR / "failure_cases"
    ENHANCED_DIR = OUTPUT_DIR / "enhanced_images"  # 新增：保存增強圖像
    FAILED_LOG = LEADERBOARD_DIR / "failed_images.txt"  # 失敗圖像日誌
    
    # 退化類型映射
    DEGRADATION_TYPE_MAP = {
        'global_blur': 'Drone Motion Blur',
        'object_blur': 'Object Motion Blur',
        'low_light': 'Low-light'
    }
    
    # LPIPS 抽樣配置（與 HenryCC 一致）
    LPIPS_SAMPLE_SIZE = 700
    LPIPS_ENABLED = True
    
    # 🎯 圖像採樣配置（評估子集）
    ENABLE_SAMPLING = False  # ✅ 評估完整 Test Split (1674 張)，不採樣
    SAMPLING_STRATEGY = "stratified"  # "stratified"（分層）或 "random"（隨機）
    SAMPLES_PER_TYPE = 500  # 分層採樣：每種退化類型的樣本數
    RANDOM_SAMPLE_RATIO = 0.10  # 隨機採樣：採樣比例（10% = 0.10）
    SAMPLING_SEED = 42  # 隨機種子（保證可重現）
    SAVE_SAMPLE_LIST = True  # 是否保存測試圖像列表（用於微調模型對比）
    SAMPLE_LIST_FILE = OUTPUT_DIR / "test_split_images.txt"  # 測試集圖像列表
    
    # Real-ESRGAN 參數（優化配置）
    TILE = 512  # 與 v1 一致，已驗證穩定（適合 RTX 3060）
    TILE_PAD = 10
    PRE_PAD = 0
    FP32 = True  # 使用 FP32（更穩定，避免內存碎片）
    OUTSCALE = 1  # 保持原大小
    
    # ⏱️ 超時配置（防止卡住）
    ENABLE_TIMEOUT = False  # ❌ 暫時禁用（有 bug，修復中）
    TIMEOUT_SECONDS = 300  # 每張圖像超時時間（5分鐘）
    
    # Checkpoint 配置
    CHECKPOINT_ENABLED = True  # 啟用 checkpoint
    CHECKPOINT_INTERVAL = 100  # 每 100 張保存一次
    CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
    CLEAR_MEMORY_AFTER_CHECKPOINT = False  # 是否在 checkpoint 後清除結果緩存（節省內存）
    
    # 保存增強圖像配置
    SAVE_ENHANCED_IMAGES = True  # 是否保存增強後的圖像
    ENHANCED_IMAGE_FORMAT = "png"  # 保存格式 (png/jpg)
    
    # 採樣配置
    NUM_SAMPLES = 50  # 對比樣本數量
    NUM_FAILURE_SAMPLES = 20
    
    # 設備
    try:
        DEVICE = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    except:
        DEVICE = 'cpu'


# ==================== Real-ESRGAN 處理器 ====================

class RealESRGANProcessor:
    """Real-ESRGAN 圖像處理器"""
    
    def __init__(self, model_path: Path, config: RealESRGANOnlyConfig):
        self.model_path = model_path
        self.config = config
        self.device = config.DEVICE
        
        print(f"\n🔧 初始化 Real-ESRGAN:")
        print(f"  模型: {model_path.name}")
        print(f"  設備: {self.device}")
        print(f"  Tile 大小: {config.TILE} (FP32 模式)")
        
        # 初始化模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=4)
        
        # 創建 Real-ESRGAN 處理器
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.model_path),
            model=model,
            tile=self.config.TILE,
            tile_pad=self.config.TILE_PAD,
            pre_pad=self.config.PRE_PAD,
            half=not self.config.FP32,  # FP32 = True → half = False
            device=self.device,
            gpu_id=None
        )
        
        print("✓ Real-ESRGAN 初始化完成")
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """增強圖像"""
        output, _ = self.upsampler.enhance(img, outscale=self.config.OUTSCALE)
        return output


# ==================== 主評估器 ====================

class RealESRGANEvaluator:
    """Real-ESRGAN 基準評估器（支持 Checkpoint）"""
    
    def __init__(self):
        self.config = RealESRGANOnlyConfig()
        
        # 檢查模型文件
        if not self.config.MODEL_PATH.exists():
            print(f"\n❌ 模型文件不存在: {self.config.MODEL_PATH}")
            print("請下載 RealESRGAN_x4plus.pth 並放置到正確位置")
            sys.exit(1)
        
        # 創建輸出目錄
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        self.config.ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        
        # 初始化處理器
        self.realesrgan = RealESRGANProcessor(self.config.MODEL_PATH, self.config)
        self.calculator = MetricsCalculator()  # MetricsCalculator 不需要參數
        
        # 載入配對信息
        self.pairs_df = self.load_pairs()
    
    def load_pairs(self) -> pd.DataFrame:
        """載入 pairs.csv"""
        if not self.config.PAIRS_CSV.exists():
            print(f"❌ 配對文件不存在: {self.config.PAIRS_CSV}")
            return pd.DataFrame()
        
        print(f"\n📁 載入配對信息: {self.config.PAIRS_CSV}")
        df = pd.read_csv(self.config.PAIRS_CSV)
        print(f"✓ 載入 {len(df)} 對配對")
        
        # 顯示退化類型統計
        print("\n退化類型統計:")
        for mode in df['mode'].unique():
            count = len(df[df['mode'] == mode])
            display_name = self.config.DEGRADATION_TYPE_MAP.get(mode, mode)
            print(f"  {display_name}: {count} 張")
        
        return df
    
    def sample_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """對圖像進行採樣（分層或隨機）"""
        # ✅ 重要：只使用 test split 的圖像進行評估
        if 'split' in df.columns:
            df_test = df[df['split'] == 'test'].copy()
            total_before = len(df)
            total_test = len(df_test)
            print(f"\n📊 Split 過濾:")
            print(f"  總圖像數: {total_before} 張")
            print(f"  Test Split: {total_test} 張 ({total_test/total_before*100:.1f}%)")
            print(f"  Train Split: {total_before - total_test} 張（不評估）")
            df = df_test
        else:
            print(f"\n⚠️  警告：pairs.csv 沒有 'split' 列，將評估所有圖像")
        
        if not self.config.ENABLE_SAMPLING:
            # 不採樣，使用全部 test split
            print(f"\n✅ 採樣模式: 關閉（評估完整 Test Split）")
            print(f"  評估圖像數: {len(df)} 張")
            
            # 保存 test split 圖像列表（用於微調模型對比）
            if self.config.SAVE_SAMPLE_LIST:
                test_images = df['degraded_path'].tolist()
                with open(self.config.SAMPLE_LIST_FILE, 'w', encoding='utf-8') as f:
                    for img_path in test_images:
                        f.write(f"{img_path}\n")
                print(f"  ✓ Test Split 圖像列表已保存: {self.config.SAMPLE_LIST_FILE.name}")
                print(f"    （可用於微調模型的相同測試集對比）")
            
            return df  # 返回全部 test split
        
        print(f"\n🎲 圖像採樣策略: {self.config.SAMPLING_STRATEGY}")
        print(f"  隨機種子: {self.config.SAMPLING_SEED}")
        
        # 設置隨機種子
        random.seed(self.config.SAMPLING_SEED)
        np.random.seed(self.config.SAMPLING_SEED)
        
        if self.config.SAMPLING_STRATEGY == "stratified":
            # 分層採樣：每種退化類型採樣固定數量
            sampled_dfs = []
            print(f"  每類樣本數: {self.config.SAMPLES_PER_TYPE}")
            print(f"\n採樣詳情:")
            
            for mode in df['mode'].unique():
                df_type = df[df['mode'] == mode]
                total_count = len(df_type)
                sample_count = min(self.config.SAMPLES_PER_TYPE, total_count)
                
                # 隨機採樣
                df_sampled = df_type.sample(n=sample_count, random_state=self.config.SAMPLING_SEED)
                sampled_dfs.append(df_sampled)
                
                display_name = self.config.DEGRADATION_TYPE_MAP.get(mode, mode)
                ratio = sample_count / total_count * 100
                print(f"  {display_name}: {sample_count}/{total_count} ({ratio:.1f}%)")
            
            df_final = pd.concat(sampled_dfs, ignore_index=True)
            
        else:  # random
            # 隨機採樣：按比例隨機採樣
            sample_count = int(len(df) * self.config.RANDOM_SAMPLE_RATIO)
            df_final = df.sample(n=sample_count, random_state=self.config.SAMPLING_SEED)
            
            print(f"  採樣比例: {self.config.RANDOM_SAMPLE_RATIO*100:.1f}%")
            print(f"  採樣數量: {sample_count}/{len(df)}")
        
        # 保存採樣列表（用於微調模型對比）
        if self.config.SAVE_SAMPLE_LIST:
            sampled_images = df_final['degraded_path'].tolist()
            with open(self.config.SAMPLE_LIST_FILE, 'w', encoding='utf-8') as f:
                for img_path in sampled_images:
                    f.write(f"{img_path}\n")
            print(f"\n✓ 採樣列表已保存: {self.config.SAMPLE_LIST_FILE.name}")
            print(f"  （可用於微調模型的相同採樣對比）")
        
        print(f"\n📊 最終 Test Split 採樣:")
        print(f"  總數: {len(df_final)}/{len(df)} ({len(df_final)/len(df)*100:.1f}%)")
        
        return df_final
    
    def save_checkpoint(self, all_results: List[Dict], processed_images: set, current_idx: int):
        """保存 checkpoint（同時保存 CSV 結果）"""
        try:
            # 1. 保存 checkpoint.json
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(self.pairs_df),
                'processed_count': len(processed_images),
                'last_index': current_idx,
                'processed_images': list(processed_images),
                'results': all_results,
                'config': {
                    'TILE': self.config.TILE,
                    'FP32': self.config.FP32,
                    'LPIPS_SAMPLE_SIZE': self.config.LPIPS_SAMPLE_SIZE
                }
            }
            
            with open(self.config.CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # 2. 同時保存 CSV 結果（即時更新）
            if all_results:
                df = pd.DataFrame(all_results)
                csv_path = self.config.LEADERBOARD_DIR / "realesrgan_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"\n⚠️  Checkpoint 保存失敗: {e}")
    
    def log_failed_image(self, image_name: str, reason: str):
        """記錄失敗的圖像到日誌文件"""
        try:
            with open(self.config.FAILED_LOG, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} | {image_name} | {reason}\n")
        except Exception as e:
            pass  # 忽略日誌寫入錯誤
    
    def load_checkpoint(self) -> tuple:
        """載入 checkpoint（支持從 V1 結果恢復）"""
        # 優先載入 checkpoint.json
        if self.config.CHECKPOINT_ENABLED and self.config.CHECKPOINT_FILE.exists():
            try:
                with open(self.config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                all_results = checkpoint_data.get('results', [])
                processed_images = set(checkpoint_data.get('processed_images', []))
                last_index = checkpoint_data.get('last_index', 0)
                
                print(f"\n✅ 已載入 Checkpoint (checkpoint.json):")
                print(f"  已處理: {len(processed_images)} 張")
                print(f"  上次索引: {last_index}")
                print(f"  跳過已完成的圖像...")
                
                return all_results, processed_images, last_index
            
            except Exception as e:
                print(f"\n⚠️  Checkpoint 載入失敗: {e}")
        
        # 備用：嘗試從現有的 CSV 結果文件恢復（V1 兼容）
        csv_path = self.config.LEADERBOARD_DIR / "realesrgan_results.csv"
        if csv_path.exists():
            try:
                print(f"\n🔄 檢測到現有結果文件，嘗試恢復...")
                df_existing = pd.read_csv(csv_path)
                
                # 從 CSV 恢復結果
                all_results = df_existing.to_dict('records')
                processed_images = set(df_existing['image_name'].unique())
                
                print(f"\n✅ 已從現有結果恢復:")
                print(f"  已處理: {len(processed_images)} 張")
                print(f"  將從此處繼續...")
                print(f"  💾 增強圖像將開始保存（從現在開始）")
                
                return all_results, processed_images, 0
            
            except Exception as e:
                print(f"\n⚠️  CSV 恢復失敗: {e}")
        
        print("\n🆕 沒有找到 checkpoint 或現有結果，從頭開始...")
        return [], set(), 0
    
    def evaluate_realesrgan(self):
        """評估 Real-ESRGAN（支持 Checkpoint 恢復 + 採樣）"""
        print("\n" + "=" * 80)
        print("🎯 開始 Real-ESRGAN 評估（Checkpoint 恢復 + 採樣支持）")
        print("=" * 80)
        
        if self.pairs_df.empty:
            print("❌ 沒有配對信息，無法進行評估")
            return
        
        # 🎲 應用採樣策略
        pairs_to_evaluate = self.sample_images(self.pairs_df)
        
        # 隨機選擇要計算 LPIPS 的圖像索引
        total_images = len(pairs_to_evaluate)
        sample_size = min(self.config.LPIPS_SAMPLE_SIZE, total_images)
        lpips_sample_indices = set(random.sample(range(total_images), sample_size))
        
        print(f"\n💡 評估策略:")
        print(f"  ✅ 採樣模式: {'啟用 (' + self.config.SAMPLING_STRATEGY + ')' if self.config.ENABLE_SAMPLING else '關閉（評估全部）'}")
        if self.config.ENABLE_SAMPLING:
            if self.config.SAMPLING_STRATEGY == "stratified":
                print(f"  ✅ 每類樣本: {self.config.SAMPLES_PER_TYPE} 張")
            else:
                print(f"  ✅ 採樣比例: {self.config.RANDOM_SAMPLE_RATIO*100:.1f}%")
        print(f"  ✅ 評估圖像: {total_images} 張")
        print(f"  ✅ PSNR/SSIM: 所有 {total_images} 張圖像")
        print(f"  ✅ LPIPS: 隨機抽樣 {sample_size} 張 ({sample_size/total_images*100:.1f}%)")
        print(f"  🔧 記憶體管理: 每張圖像後清理 GPU 快取")
        print(f"  💾 Checkpoint: 每 {self.config.CHECKPOINT_INTERVAL} 張保存")
        print(f"  📸 增強圖像: {'啟用保存' if self.config.SAVE_ENHANCED_IMAGES else '不保存'} ({self.config.ENHANCED_IMAGE_FORMAT})")
        print(f"  ⚡ 使用設備: {self.config.DEVICE}")
        print(f"  🎛️  TILE: {self.config.TILE} (FP32 穩定模式)")
        print(f"  ⏱️  超時保護: {'啟用' if self.config.ENABLE_TIMEOUT else '停用'} ({self.config.TIMEOUT_SECONDS if self.config.ENABLE_TIMEOUT else 'N/A'} 秒/張)")
        
        # 預估時間
        if self.config.ENABLE_SAMPLING and total_images < 2000:
            estimated_hours = total_images * 90 / 3600  # ~90秒/張
            print(f"\n📈 預計時間: ~{estimated_hours:.1f} 小時（採樣加速）")
        else:
            print(f"\n📈 預計時間: ~30-35 小時（全量評估）")
        
        # 載入 checkpoint
        all_results, processed_images, start_idx = self.load_checkpoint()
        
        # 處理每對圖像（顯示實際要處理的數量）
        print(f"\n📊 處理 {len(pairs_to_evaluate)} 張圖像...")
        
        failed_count = 0
        success_count = len(processed_images)
        
        for enum_idx, (idx, row) in enumerate(tqdm(pairs_to_evaluate.iterrows(), 
                                                    total=len(pairs_to_evaluate), 
                                                    desc="Real-ESRGAN 評估",
                                                    initial=start_idx)):
            degraded_path = Path(row['degraded_path'])
            
            # 跳過已處理的圖像
            if degraded_path.name in processed_images:
                continue
            
            original_path = Path(row['target_path'])
            deg_type = row['mode']
            
            # 檢查文件是否存在
            if not degraded_path.exists():
                print(f"\n  ⚠️  退化圖像不存在: {degraded_path.name}")
                continue
            
            if not original_path.exists():
                print(f"\n  ⚠️  原始圖像不存在: {original_path.name}")
                continue
            
            # 讀取圖像
            degraded = cv2.imread(str(degraded_path))
            original = cv2.imread(str(original_path))
            
            if degraded is None or original is None:
                continue
            
            # 確保尺寸一致
            if degraded.shape != original.shape:
                degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
            
            # 判斷是否對這張圖計算 LPIPS
            calculate_lpips_for_this_image = enum_idx in lpips_sample_indices
            
            try:
                # ⏱️ 使用超時機制（如果啟用）
                if self.config.ENABLE_TIMEOUT:
                    with time_limit(self.config.TIMEOUT_SECONDS) as timer:
                        # 應用 Real-ESRGAN
                        enhanced = self.realesrgan.enhance(degraded)
                        
                        # 檢查是否超時
                        if timer.timed_out:
                            raise TimeoutException(f"處理超時（>{self.config.TIMEOUT_SECONDS}秒）")
                        
                        # 計算有參考指標
                        metrics = self.calculator.evaluate_with_reference(
                            enhanced, original, 
                            calculate_lpips=calculate_lpips_for_this_image
                        )
                else:
                    # 不使用超時（原始行為）
                    enhanced = self.realesrgan.enhance(degraded)
                    metrics = self.calculator.evaluate_with_reference(
                        enhanced, original, 
                        calculate_lpips=calculate_lpips_for_this_image
                    )
                
                # 💾 保存增強圖像（如果啟用）
                if self.config.SAVE_ENHANCED_IMAGES:
                    enhanced_filename = f"enhanced_{degraded_path.stem}.{self.config.ENHANCED_IMAGE_FORMAT}"
                    enhanced_save_path = self.config.ENHANCED_DIR / enhanced_filename
                    
                    # 保存圖像
                    if self.config.ENHANCED_IMAGE_FORMAT == 'png':
                        cv2.imwrite(str(enhanced_save_path), enhanced, 
                                   [cv2.IMWRITE_PNG_COMPRESSION, 3])  # 0-9, 3=好的壓縮
                    else:  # jpg
                        cv2.imwrite(str(enhanced_save_path), enhanced, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])  # 95=高質量
                
                # 記錄結果
                result_entry = {
                    'image_name': degraded_path.name,
                    'degradation_type': self.config.DEGRADATION_TYPE_MAP.get(deg_type, deg_type),
                    'degradation_mode': deg_type,
                    'method': 'realesrgan',
                    'split': row['split'],
                    **metrics
                }
                all_results.append(result_entry)
                processed_images.add(degraded_path.name)
                success_count += 1
                
                # 🎯 首次成功立即保存（讓用戶看到進展）
                if success_count == 1:
                    self.save_checkpoint(all_results, processed_images, enum_idx + 1)
                    print(f"\n💾 首次結果已保存！後續每 {self.config.CHECKPOINT_INTERVAL} 張保存一次")
                
            except TimeoutException as e:
                # 超時異常處理
                print(f"\n  ⏱️  超時跳過: {degraded_path.name} - {str(e)}")
                self.log_failed_image(degraded_path.name, f"TIMEOUT: {str(e)}")
                failed_count += 1
                
                # 強制清理內存
                try:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        for _ in range(3):
                            try:
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                gc.collect()
                            except:
                                pass
                except:
                    pass
                
                processed_images.add(degraded_path.name)
                continue
                
            except Exception as e:
                # 錯誤處理：記錄失敗並繼續
                error_msg = str(e)
                if 'out of memory' in error_msg.lower() or 'oom' in error_msg.lower():
                    print(f"\n  ⚠️  GPU OOM: {degraded_path.name} - 跳過並清理內存")
                    self.log_failed_image(degraded_path.name, f"OOM: {error_msg[:50]}")
                    failed_count += 1
                    
                    # 強制清理內存（多次嘗試）
                    try:
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            # 嘗試多次清理
                            for _ in range(3):
                                try:
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()
                                    gc.collect()
                                except:
                                    pass
                            # 等待 GPU 穩定
                            try:
                                torch.cuda.synchronize()
                            except:
                                pass
                    except:
                        pass
                else:
                    print(f"\n  ⚠️  處理失敗: {degraded_path.name} - {error_msg[:80]}")
                    self.log_failed_image(degraded_path.name, f"ERROR: {error_msg[:50]}")
                    failed_count += 1
                
                # 即使失敗也標記為已處理（避免無限重試）
                processed_images.add(degraded_path.name)
                continue
            
            finally:
                # 每張圖像後立即清理內存（安全模式）
                try:
                    if 'degraded' in locals():
                        del degraded
                    if 'original' in locals():
                        del original
                    if 'enhanced' in locals():
                        del enhanced
                except:
                    pass
                
                # 嘗試清理 GPU 緩存（忽略所有錯誤）
                try:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass  # 忽略清理時的任何錯誤
            
            # 定期保存 checkpoint + CSV（基于成功+失败总数）
            total_processed = success_count + failed_count
            if total_processed > 0 and total_processed % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(all_results, processed_images, enum_idx + 1)
                csv_path = self.config.LEADERBOARD_DIR / "realesrgan_results.csv"
                print(f"\n💾 已保存 Checkpoint #{total_processed // self.config.CHECKPOINT_INTERVAL}")
                print(f"   ├─ 成功: {success_count} 張")
                print(f"   ├─ 失敗: {failed_count} 張")
                print(f"   ├─ Checkpoint: {self.config.CHECKPOINT_FILE.name}")
                print(f"   └─ CSV: {csv_path.name} ({len(all_results)} 筆成功記錄)")
                
                # 可選：清除內存緩存（節省 CPU RAM）
                if self.config.CLEAR_MEMORY_AFTER_CHECKPOINT:
                    all_results.clear()  # 清除列表，但 CSV 已保存
                    print(f"   └─ 內存緩存已清除（節省 RAM）")
        
        # 最終保存
        self.save_checkpoint(all_results, processed_images, len(pairs_to_evaluate))
        
        # 如果啟用了內存清理，從 CSV 讀取最終統計
        csv_path = self.config.LEADERBOARD_DIR / "realesrgan_results.csv"
        if self.config.CLEAR_MEMORY_AFTER_CHECKPOINT and csv_path.exists():
            try:
                df_final = pd.read_csv(csv_path)
                final_count = len(df_final)
            except:
                final_count = len(all_results)
        else:
            final_count = len(all_results)
        
        if final_count == 0:
            print("\n❌ 沒有生成任何結果")
            return
        
        print(f"\n📊 評估完成:")
        print(f"  ✅ 成功: {success_count} 張")
        print(f"  ❌ 失敗: {failed_count} 張")
        print(f"  📈 成功率: {success_count/(success_count+failed_count)*100:.1f}%")
        print(f"  📝 結果記錄: {final_count} 筆")
        
        # CSV 已在 save_checkpoint 中保存
        print(f"\n✓ 完整結果已保存: {csv_path}")
        print(f"✓ Checkpoint 已保存: {self.config.CHECKPOINT_FILE}")
        
        # 生成排行榜（與 HenryCC baseline 相同格式）
        if csv_path.exists():
            print(f"\n📊 生成排行榜...")
            try:
                df_results = pd.read_csv(csv_path)
                self.generate_leaderboards(df_results)
            except Exception as e:
                print(f"⚠️  排行榜生成失敗: {e}")
        
        print("\n✅ Real-ESRGAN 評估完成！")
    
    def generate_leaderboards(self, df: pd.DataFrame):
        """生成排行榜（與 baseline_evaluation_with_gt-HenryCC.py 相同邏輯）"""
        print("  生成各退化類型的排行榜...")
        
        # 按退化類型分組
        for deg_type in df['degradation_type'].unique():
            df_type = df[df['degradation_type'] == deg_type]
            
            # 計算平均指標
            agg_dict = {
                'var_laplacian': 'mean',
                'tenengrad': 'mean',
                'blur_extent': 'mean',
                'luminance': 'mean',
                'contrast': 'mean',
                'entropy': 'mean',
                'gradient_mean': 'mean',
            }
            
            # 添加有參考指標
            if 'psnr' in df_type.columns:
                agg_dict['psnr'] = 'mean'
            if 'ssim' in df_type.columns:
                agg_dict['ssim'] = 'mean'
            if 'lpips' in df_type.columns:
                # 只計算非 NaN 值的平均
                agg_dict['lpips'] = 'mean'
            
            leaderboard = df_type.groupby('method').agg(agg_dict).round(4)
            
            # 計算綜合得分（與 HenryCC baseline 相同公式）
            # 1. PSNR/SSIM 得分（如果有）
            if 'psnr' in leaderboard.columns and 'ssim' in leaderboard.columns:
                # 歸一化 PSNR（假設範圍 20-40 dB）
                psnr_normalized = (leaderboard['psnr'] - 20) / 20
                psnr_normalized = psnr_normalized.clip(0, 1)
                
                # SSIM 已經在 0-1 範圍
                ssim_normalized = leaderboard['ssim']
                
                # 綜合得分（PSNR 50% + SSIM 50%）
                leaderboard['reference_score'] = (psnr_normalized + ssim_normalized) / 2
            
            # 2. 銳度得分
            leaderboard['sharpness_score'] = (
                leaderboard['var_laplacian'] / leaderboard['var_laplacian'].max() +
                leaderboard['tenengrad'] / leaderboard['tenengrad'].max()
            ) / 2
            
            # 3. 模糊得分
            leaderboard['blur_score'] = 1 - (leaderboard['blur_extent'] / leaderboard['blur_extent'].max())
            
            # 4. 總體得分
            if 'reference_score' in leaderboard.columns:
                # 如果有參考指標，參考得分佔 60%，其他佔 40%
                leaderboard['overall_score'] = (
                    0.6 * leaderboard['reference_score'] +
                    0.2 * leaderboard['sharpness_score'] +
                    0.2 * leaderboard['blur_score']
                )
            else:
                # 只有無參考指標
                leaderboard['overall_score'] = (
                    leaderboard['sharpness_score'] + leaderboard['blur_score']
                ) / 2
            
            # 排序
            leaderboard = leaderboard.sort_values('overall_score', ascending=False)
            
            # 保存
            deg_type_safe = deg_type.replace(' ', '_').replace('-', '_').lower()
            output_path = self.config.LEADERBOARD_DIR / f"leaderboard_{deg_type_safe}.csv"
            leaderboard.to_csv(output_path, encoding='utf-8-sig')
            print(f"    ✓ {deg_type} 排行榜已保存: {output_path.name}")
            
            # 打印分數
            print(f"    🏆 {deg_type} - Real-ESRGAN:")
            for method, row in leaderboard.iterrows():
                psnr_str = f"PSNR: {row['psnr']:.2f} dB, " if 'psnr' in row else ""
                ssim_str = f"SSIM: {row['ssim']:.4f}, " if 'ssim' in row else ""
                lpips_str = f"LPIPS: {row['lpips']:.4f}, " if 'lpips' in row and not pd.isna(row['lpips']) else ""
                print(f"       {psnr_str}{ssim_str}{lpips_str}Overall: {row['overall_score']:.4f}")


# ==================== 主程序 ====================

def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("🚀 Real-ESRGAN 基準評估（單獨運行 v2 - Checkpoint 支持）")
    print("=" * 80)
    print("\n📋 配置資訊:")
    print(f"  退化圖像目錄: {RealESRGANOnlyConfig.DEGRADED_DIR}")
    print(f"  原始圖像目錄: {RealESRGANOnlyConfig.ORIGINAL_DIR}")
    print(f"  配對文件: {RealESRGANOnlyConfig.PAIRS_CSV}")
    print(f"  輸出目錄: {RealESRGANOnlyConfig.OUTPUT_DIR}")
    print(f"  設備: {RealESRGANOnlyConfig.DEVICE}")
    print(f"  TILE 大小: {RealESRGANOnlyConfig.TILE} (FP32)")
    print(f"  Checkpoint: {'啟用' if RealESRGANOnlyConfig.CHECKPOINT_ENABLED else '停用'}")
    
    # 創建評估器
    evaluator = RealESRGANEvaluator()
    
    # 執行評估
    evaluator.evaluate_realesrgan()
    
    print("\n✅ 所有任務完成！")
    print(f"\n📁 請檢查輸出目錄: {RealESRGANOnlyConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

