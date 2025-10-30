#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料準備腳本 (Data Preparation Script)
用途：為 Real-ESRGAN Fine-tuning 準備高品質訓練資料

步驟：
1. 分析影像品質和多樣性
2. 偵測並標記重複/相似影像
3. 資料增強（翻轉、旋轉、裁切）

日期：2025-10-29
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json
from typing import List, Dict, Tuple
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置區 ====================

# 資料夾路徑
FYP_DIR = Path("/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP")
IMAGES_BASE_DIR = Path("/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP_Images")

# CSV 檔案列表（排除 backup）
CSV_FILES = [
    "accepted_set_Aerial-Traffic.csv",
    "accepted_set_FloodNet.csv",
    "accepted_set_UAV-VisLoc.csv",
    "accepted_set_VisDrone2019.csv",
    "accepted_set_SwissOkutama.csv"
]

# 資料夾對應
DATASET_FOLDERS = {
    "Aerial-Traffic": "Images_Aerial-Traffic",
    "FloodNet": "Images_FloodNet",
    "UAV-VisLoc": "Images_UAV-VisLoc",
    "VisDrone2019": "Images_VisDrone2019",
    "SwissOkutama": "Images_SwissOkutama"
}

# 資料增強參數
AUGMENTATION_CONFIG = {
    "enable_flip": True,           # 翻轉（水平）
    "enable_rotation": False,      # 旋轉（關閉以降低擴充量）
    "enable_crop": True,           # 裁切
    "crop_size": 512,              # 裁切大小
    "num_crops_per_image": 2,      # 每張影像裁切數量（降低至 2）
    "min_resolution": 512          # 最小解析度（低於此值不做裁切）
}

# 相似度檢測參數
SIMILARITY_CONFIG = {
    "phash_threshold": 5,          # Perceptual hash 差異閾值（越小越相似）
    "ssim_threshold": 0.95,        # SSIM 相似度閾值（越高越相似）
    "enable_phash": True,          # 啟用感知雜湊
    "enable_feature": False        # 啟用特徵比對（較慢但更準確）
}

# 輸出資料夾
OUTPUT_DIR = FYP_DIR / "data_preparation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== 步驟 1：分析影像品質和多樣性 ====================

class ImageQualityAnalyzer:
    """影像品質與多樣性分析器"""
    
    def __init__(self):
        self.results = []
        
    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        計算 Laplacian variance（用於評估影像清晰度）
        值越高表示影像越清晰
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    
    def calculate_brightness_stats(self, image: np.ndarray) -> Dict:
        """計算亮度統計資訊"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return {
            "mean_brightness": float(gray.mean()),
            "std_brightness": float(gray.std()),
            "min_brightness": int(gray.min()),
            "max_brightness": int(gray.max())
        }
    
    def calculate_edge_density(self, image: np.ndarray) -> float:
        """
        計算邊緣密度（用於評估影像細節豐富度）
        使用 Canny 邊緣檢測
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return float(edge_density)
    
    def analyze_image(self, image_path: str, metadata: Dict = None) -> Dict:
        """分析單張影像"""
        try:
            # 讀取影像
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "無法讀取影像", "filepath": image_path}
            
            # 基本資訊
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            
            # 品質指標
            laplacian_var = self.calculate_laplacian_variance(img)
            brightness_stats = self.calculate_brightness_stats(img)
            edge_density = self.calculate_edge_density(img)
            
            # 檔案資訊
            file_size = os.path.getsize(image_path)
            
            result = {
                "filepath": image_path,
                "filename": os.path.basename(image_path),
                "width": width,
                "height": height,
                "channels": channels,
                "resolution": f"{width}x{height}",
                "total_pixels": width * height,
                "file_size_mb": file_size / (1024 * 1024),
                "laplacian_variance": laplacian_var,
                "edge_density": edge_density,
                **brightness_stats
            }
            
            # 加入 metadata（從 CSV）
            if metadata:
                result.update({
                    "scene_type": metadata.get("scene_type", ""),
                    "has_text": metadata.get("has_text", 0),
                    "is_night": metadata.get("is_night", 0),
                    "quality_level": metadata.get("quality_level", "")
                })
            
            return result
            
        except Exception as e:
            return {"error": str(e), "filepath": image_path}
    
    def analyze_dataset(self, image_paths: List[str], metadata_dict: Dict = None) -> pd.DataFrame:
        """分析整個資料集"""
        print(f"📊 分析 {len(image_paths)} 張影像...")
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 50 == 0:
                print(f"  進度: {i+1}/{len(image_paths)}")
            
            metadata = metadata_dict.get(img_path, {}) if metadata_dict else None
            result = self.analyze_image(img_path, metadata)
            self.results.append(result)
        
        df = pd.DataFrame(self.results)
        return df
    
    def generate_diversity_report(self, df: pd.DataFrame) -> Dict:
        """生成多樣性報告"""
        # 處理空資料夾情況
        if len(df) == 0 or df.empty:
            return {
                "total_images": 0,
                "resolution_distribution": {},
                "scene_type_distribution": {},
                "brightness_stats": {"mean": 0, "std": 0, "range": [0, 0]},
                "quality_metrics": {"avg_laplacian_variance": 0, "avg_edge_density": 0, "sharpness_range": [0, 0]}
            }
        
        report = {
            "total_images": len(df),
            "resolution_distribution": df["resolution"].value_counts().to_dict() if "resolution" in df.columns else {},
            "scene_type_distribution": {},
            "brightness_stats": {
                "mean": float(df["mean_brightness"].mean()) if "mean_brightness" in df.columns else 0,
                "std": float(df["mean_brightness"].std()) if "mean_brightness" in df.columns else 0,
                "range": [float(df["mean_brightness"].min()), float(df["mean_brightness"].max())] if "mean_brightness" in df.columns else [0, 0]
            },
            "quality_metrics": {
                "avg_laplacian_variance": float(df["laplacian_variance"].mean()) if "laplacian_variance" in df.columns else 0,
                "avg_edge_density": float(df["edge_density"].mean()) if "edge_density" in df.columns else 0,
                "sharpness_range": [float(df["laplacian_variance"].min()), float(df["laplacian_variance"].max())] if "laplacian_variance" in df.columns else [0, 0]
            }
        }
        
        # 場景類型分佈
        if "scene_type" in df.columns:
            all_scenes = []
            for scenes in df["scene_type"].dropna():
                if isinstance(scenes, str):
                    all_scenes.extend([s.strip() for s in scenes.split(",")])
            from collections import Counter
            report["scene_type_distribution"] = dict(Counter(all_scenes))
        
        # 夜間影像比例
        if "is_night" in df.columns:
            report["night_image_ratio"] = float(df["is_night"].mean())
        
        return report


# ==================== 步驟 2：偵測重複/相似影像 ====================

class SimilarityDetector:
    """相似影像偵測器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.phash_dict = {}
        
    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> str:
        """
        計算 Perceptual Hash (感知雜湊)
        相似的影像會產生相似的 hash
        """
        # 縮放到小尺寸
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        
        # 轉換為灰階
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        
        # 計算差分（DCT-based perceptual hash）
        diff = gray[:, 1:] > gray[:, :-1]
        
        # 轉換為 hash 字串
        hash_str = ''.join(['1' if x else '0' for row in diff for x in row])
        return hash_str
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """計算兩個 hash 的 Hamming distance"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def find_similar_images(self, image_paths: List[str]) -> List[Dict]:
        """找出相似的影像對"""
        print(f"🔍 偵測相似影像（共 {len(image_paths)} 張）...")
        
        # 計算所有影像的 phash
        phashes = {}
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"  計算 hash: {i+1}/{len(image_paths)}")
            
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    phash = self.compute_phash(img)
                    phashes[img_path] = phash
            except Exception as e:
                print(f"  ⚠️  無法處理 {img_path}: {e}")
        
        # 比對相似度
        similar_pairs = []
        paths = list(phashes.keys())
        
        print(f"  開始比對相似度...")
        for i in range(len(paths)):
            if (i + 1) % 50 == 0:
                print(f"  比對進度: {i+1}/{len(paths)}")
            
            for j in range(i + 1, len(paths)):
                hash1 = phashes[paths[i]]
                hash2 = phashes[paths[j]]
                distance = self.hamming_distance(hash1, hash2)
                
                if distance <= self.config["phash_threshold"]:
                    similar_pairs.append({
                        "image1": paths[i],
                        "image2": paths[j],
                        "hamming_distance": distance,
                        "similarity_score": 1 - (distance / len(hash1))
                    })
        
        return similar_pairs
    
    def generate_removal_suggestions(self, similar_pairs: List[Dict]) -> List[str]:
        """生成建議移除的影像清單"""
        # 統計每張影像出現在相似對中的次數
        image_count = defaultdict(int)
        for pair in similar_pairs:
            image_count[pair["image1"]] += 1
            image_count[pair["image2"]] += 1
        
        # 建議移除：在相似對中出現最多次的影像
        suggestions = []
        processed = set()
        
        for pair in sorted(similar_pairs, key=lambda x: x["similarity_score"], reverse=True):
            img1, img2 = pair["image1"], pair["image2"]
            
            if img1 not in processed and img2 not in processed:
                # 保留出現次數較少的那張（表示它比較獨特）
                if image_count[img1] >= image_count[img2]:
                    suggestions.append(img1)
                    processed.add(img1)
                else:
                    suggestions.append(img2)
                    processed.add(img2)
        
        return suggestions


# ==================== 步驟 3：資料增強 ====================

class DataAugmentor:
    """資料增強器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_log = []
        
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """水平翻轉"""
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """垂直翻轉"""
        return cv2.flip(image, 0)
    
    def rotate_90(self, image: np.ndarray) -> np.ndarray:
        """旋轉 90 度"""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def rotate_180(self, image: np.ndarray) -> np.ndarray:
        """旋轉 180 度"""
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def rotate_270(self, image: np.ndarray) -> np.ndarray:
        """旋轉 270 度"""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def random_crop(self, image: np.ndarray, crop_size: int) -> np.ndarray:
        """隨機裁切"""
        h, w = image.shape[:2]
        
        if h < crop_size or w < crop_size:
            return None
        
        # 隨機選擇起始點
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)
        
        return image[top:top+crop_size, left:left+crop_size]
    
    def augment_single_image(self, image_path: str, output_dir: str) -> List[str]:
        """
        對單張影像執行資料增強
        返回所有生成的影像路徑
        """
        try:
            # 讀取影像
            img = cv2.imread(image_path)
            if img is None:
                print(f"  ⚠️  無法讀取: {image_path}")
                return []
            
            h, w = img.shape[:2]
            basename = Path(image_path).stem
            ext = Path(image_path).suffix
            
            generated_paths = []
            
            # 1. 翻轉增強（只做水平翻轉）
            if self.config["enable_flip"]:
                # 水平翻轉
                flipped_h = self.flip_horizontal(img)
                output_path = os.path.join(output_dir, f"{basename}_flip_h{ext}")
                cv2.imwrite(output_path, flipped_h)
                generated_paths.append(output_path)
                self.augmentation_log.append({
                    "original": image_path,
                    "augmented": output_path,
                    "method": "horizontal_flip"
                })
            
            # 2. 旋轉增強
            if self.config["enable_rotation"]:
                # 90 度
                rotated_90 = self.rotate_90(img)
                output_path = os.path.join(output_dir, f"{basename}_rot90{ext}")
                cv2.imwrite(output_path, rotated_90)
                generated_paths.append(output_path)
                self.augmentation_log.append({
                    "original": image_path,
                    "augmented": output_path,
                    "method": "rotate_90"
                })
                
                # 180 度
                rotated_180 = self.rotate_180(img)
                output_path = os.path.join(output_dir, f"{basename}_rot180{ext}")
                cv2.imwrite(output_path, rotated_180)
                generated_paths.append(output_path)
                self.augmentation_log.append({
                    "original": image_path,
                    "augmented": output_path,
                    "method": "rotate_180"
                })
                
                # 270 度
                rotated_270 = self.rotate_270(img)
                output_path = os.path.join(output_dir, f"{basename}_rot270{ext}")
                cv2.imwrite(output_path, rotated_270)
                generated_paths.append(output_path)
                self.augmentation_log.append({
                    "original": image_path,
                    "augmented": output_path,
                    "method": "rotate_270"
                })
            
            # 3. 裁切增強（只對足夠大的影像）
            if self.config["enable_crop"] and min(h, w) >= self.config["crop_size"]:
                for i in range(self.config["num_crops_per_image"]):
                    cropped = self.random_crop(img, self.config["crop_size"])
                    if cropped is not None:
                        output_path = os.path.join(output_dir, f"{basename}_crop{i}{ext}")
                        cv2.imwrite(output_path, cropped)
                        generated_paths.append(output_path)
                        self.augmentation_log.append({
                            "original": image_path,
                            "augmented": output_path,
                            "method": f"random_crop_{i}"
                        })
            
            return generated_paths
            
        except Exception as e:
            print(f"  ❌ 增強失敗 {image_path}: {e}")
            return []
    
    def augment_dataset(self, image_paths: List[str], dataset_name: str, metadata_df: pd.DataFrame = None) -> Dict:
        """對整個資料集執行資料增強"""
        print(f"\n🔄 開始資料增強: {dataset_name}")
        print(f"  原始影像數量: {len(image_paths)}")
        
        total_generated = 0
        augmented_metadata = []
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 50 == 0:
                print(f"  進度: {i+1}/{len(image_paths)} (已生成 {total_generated} 張)")
            
            # 確定輸出資料夾（與原始影像同一個資料夾）
            output_dir = os.path.dirname(img_path)
            
            # 執行增強
            generated = self.augment_single_image(img_path, output_dir)
            total_generated += len(generated)
            
            # 為增強影像生成 metadata（繼承原始影像的標籤）
            if metadata_df is not None and len(generated) > 0:
                # 找到原始影像的 metadata
                original_meta = metadata_df[metadata_df['filepath'] == img_path]
                if not original_meta.empty:
                    original_row = original_meta.iloc[0].to_dict()
                    
                    # 為每個增強影像創建 metadata
                    for aug_path in generated:
                        aug_meta = original_row.copy()
                        aug_meta['filepath'] = aug_path
                        aug_meta['timestamp'] = datetime.now().isoformat()
                        # 添加標記表示這是增強影像
                        aug_meta['is_augmented'] = True
                        aug_meta['original_image'] = img_path
                        augmented_metadata.append(aug_meta)
        
        print(f"  ✅ 完成！生成 {total_generated} 張增強影像")
        
        return {
            "dataset": dataset_name,
            "original_count": len(image_paths),
            "augmented_count": total_generated,
            "total_count": len(image_paths) + total_generated,
            "augmented_metadata": augmented_metadata
        }


# ==================== 主程式 ====================

def load_csv_metadata() -> Dict[str, pd.DataFrame]:
    """載入所有 CSV 檔案"""
    print("📁 載入 CSV 元數據...")
    metadata_dfs = {}
    
    for csv_file in CSV_FILES:
        csv_path = FYP_DIR / csv_file
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dataset_name = csv_file.replace("accepted_set_", "").replace(".csv", "")
            metadata_dfs[dataset_name] = df
            print(f"  ✓ {dataset_name}: {len(df)} 張影像")
        else:
            print(f"  ⚠️  找不到: {csv_file}")
    
    return metadata_dfs


def get_image_paths(metadata_dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """獲取所有影像路徑（排除 rejected_images）"""
    print("\n📸 收集影像路徑...")
    image_paths_dict = {}
    
    for dataset_name, df in metadata_dfs.items():
        folder_name = DATASET_FOLDERS.get(dataset_name)
        if not folder_name:
            continue
        
        folder_path = IMAGES_BASE_DIR / folder_name
        
        # 從 CSV 獲取路徑，但過濾掉 rejected_images
        valid_paths = []
        for filepath in df["filepath"]:
            if "rejected_images" in filepath:
                continue
            
            # 修正路徑：CSV 中可能記錄的是舊路徑 (FYP/Images_xxx)
            # 實際路徑應該是 (FYP_Images/Images_xxx)
            if not os.path.exists(filepath):
                # 嘗試修正路徑
                fixed_path = filepath.replace("/FYP/Images_", "/FYP_Images/Images_")
                if os.path.exists(fixed_path):
                    valid_paths.append(fixed_path)
                else:
                    # 如果還是找不到，嘗試用檔名重建路徑
                    filename = os.path.basename(filepath)
                    reconstructed_path = folder_path / filename
                    if os.path.exists(reconstructed_path):
                        valid_paths.append(str(reconstructed_path))
            else:
                valid_paths.append(filepath)
        
        image_paths_dict[dataset_name] = valid_paths
        print(f"  ✓ {dataset_name}: {len(valid_paths)} 張有效影像")
    
    return image_paths_dict


def main():
    """主函數"""
    print("=" * 80)
    print("🚀 資料準備流程開始")
    print("=" * 80)
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("⚠️  重要提示：")
    print("  ✅ 本腳本不會刪除任何影像檔案")
    print("  ✅ 步驟 1-2 只會分析和生成建議報告")
    print("  ✅ 步驟 3 會在原資料夾中生成增強影像")
    print("  ✅ 所有原始影像都會保留\n")
    
    # 載入元數據
    metadata_dfs = load_csv_metadata()
    
    # 獲取影像路徑
    image_paths_dict = get_image_paths(metadata_dfs)
    
    # 統計總數
    total_images = sum(len(paths) for paths in image_paths_dict.values())
    print(f"\n📊 總共 {total_images} 張影像待處理\n")
    
    # ==================== 步驟 1：品質分析 ====================
    print("\n" + "=" * 80)
    print("步驟 1：影像品質與多樣性分析")
    print("=" * 80)
    
    all_analysis_results = []
    diversity_reports = {}
    
    for dataset_name, image_paths in image_paths_dict.items():
        print(f"\n▶ 分析資料集: {dataset_name}")
        
        # 建立 metadata 字典（用於快速查找）
        metadata_dict = {}
        if dataset_name in metadata_dfs:
            df = metadata_dfs[dataset_name]
            for _, row in df.iterrows():
                metadata_dict[row["filepath"]] = row.to_dict()
        
        # 執行分析
        analyzer = ImageQualityAnalyzer()
        results_df = analyzer.analyze_dataset(image_paths, metadata_dict)
        
        # 生成多樣性報告
        diversity_report = analyzer.generate_diversity_report(results_df)
        diversity_reports[dataset_name] = diversity_report
        
        # 保存結果
        output_csv = OUTPUT_DIR / f"quality_analysis_{dataset_name}.csv"
        results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"  ✓ 分析結果已保存: {output_csv}")
        
        all_analysis_results.append(results_df)
    
    # 合併所有結果
    combined_df = pd.concat(all_analysis_results, ignore_index=True)
    combined_output = OUTPUT_DIR / "quality_analysis_all.csv"
    combined_df.to_csv(combined_output, index=False, encoding='utf-8-sig')
    print(f"\n✅ 所有分析結果已保存: {combined_output}")
    
    # 保存多樣性報告
    diversity_output = OUTPUT_DIR / "diversity_report.json"
    with open(diversity_output, 'w', encoding='utf-8') as f:
        json.dump(diversity_reports, f, indent=2, ensure_ascii=False)
    print(f"✅ 多樣性報告已保存: {diversity_output}")
    
    # ==================== 步驟 2：相似度檢測 ====================
    print("\n" + "=" * 80)
    print("步驟 2：偵測重複/相似影像")
    print("=" * 80)
    
    similarity_results = {}
    all_removal_suggestions = []
    
    for dataset_name, image_paths in image_paths_dict.items():
        print(f"\n▶ 檢測資料集: {dataset_name}")
        
        detector = SimilarityDetector(SIMILARITY_CONFIG)
        similar_pairs = detector.find_similar_images(image_paths)
        
        print(f"  發現 {len(similar_pairs)} 組相似影像對")
        
        if len(similar_pairs) > 0:
            # 生成移除建議
            suggestions = detector.generate_removal_suggestions(similar_pairs)
            print(f"  建議移除 {len(suggestions)} 張重複影像")
            
            # 保存結果
            similar_pairs_df = pd.DataFrame(similar_pairs)
            output_csv = OUTPUT_DIR / f"similar_pairs_{dataset_name}.csv"
            similar_pairs_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"  ✓ 相似影像對已保存: {output_csv}")
            
            suggestions_df = pd.DataFrame({"suggested_removal": suggestions})
            suggestions_output = OUTPUT_DIR / f"removal_suggestions_{dataset_name}.csv"
            suggestions_df.to_csv(suggestions_output, index=False, encoding='utf-8-sig')
            print(f"  ✓ 移除建議已保存: {suggestions_output}")
            
            similarity_results[dataset_name] = {
                "similar_pairs_count": len(similar_pairs),
                "suggested_removals": len(suggestions)
            }
            
            all_removal_suggestions.extend(suggestions)
    
    # 保存總結
    similarity_summary = OUTPUT_DIR / "similarity_summary.json"
    with open(similarity_summary, 'w', encoding='utf-8') as f:
        json.dump(similarity_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 相似度檢測總結已保存: {similarity_summary}")
    print(f"📊 總共建議移除 {len(all_removal_suggestions)} 張重複影像")
    
    # ==================== 步驟 3：資料增強 ====================
    print("\n" + "=" * 80)
    print("步驟 3：資料增強")
    print("=" * 80)
    
    augmentation_results = []
    all_augmented_metadata = []
    
    for dataset_name, image_paths in image_paths_dict.items():
        # 獲取對應的 metadata
        metadata_df = metadata_dfs.get(dataset_name)
        
        augmentor = DataAugmentor(AUGMENTATION_CONFIG)
        result = augmentor.augment_dataset(image_paths, dataset_name, metadata_df)
        augmentation_results.append(result)
        
        # 保存增強日誌
        log_output = OUTPUT_DIR / f"augmentation_log_{dataset_name}.csv"
        log_df = pd.DataFrame(augmentor.augmentation_log)
        log_df.to_csv(log_output, index=False, encoding='utf-8-sig')
        print(f"  ✓ 增強日誌已保存: {log_output}")
        
        # 保存增強影像的 metadata（繼承原始影像的標籤）
        if 'augmented_metadata' in result and len(result['augmented_metadata']) > 0:
            aug_meta_df = pd.DataFrame(result['augmented_metadata'])
            aug_meta_output = OUTPUT_DIR / f"augmented_metadata_{dataset_name}.csv"
            aug_meta_df.to_csv(aug_meta_output, index=False, encoding='utf-8-sig')
            print(f"  ✓ 增強影像 metadata 已保存: {aug_meta_output}")
            
            all_augmented_metadata.extend(result['augmented_metadata'])
    
    # 保存增強總結
    augmentation_summary = OUTPUT_DIR / "augmentation_summary.json"
    with open(augmentation_summary, 'w', encoding='utf-8') as f:
        json.dump(augmentation_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 資料增強總結已保存: {augmentation_summary}")
    
    # ==================== 最終總結 ====================
    print("\n" + "=" * 80)
    print("🎉 資料準備流程完成！")
    print("=" * 80)
    
    total_original = sum(r["original_count"] for r in augmentation_results)
    total_augmented = sum(r["augmented_count"] for r in augmentation_results)
    total_final = sum(r["total_count"] for r in augmentation_results)
    
    print(f"\n📊 最終統計：")
    print(f"  原始影像數量: {total_original}")
    print(f"  增強影像數量: {total_augmented}")
    print(f"  總影像數量: {total_final}")
    print(f"  擴充倍率: {total_final / total_original:.2f}x")
    print(f"\n📁 所有結果已保存至: {OUTPUT_DIR}")
    print(f"\n⏰ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

