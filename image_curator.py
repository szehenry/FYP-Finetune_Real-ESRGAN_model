#!/usr/bin/env python3
"""
圖像篩選和標註工具
用於Real-ESRGAN項目的數據集準備

功能：
- 遞歸掃描文件夾中的圖像
- 交互式圖像查看和標註
- 支持接受/拒絕、文本標籤、場景類型標註
- 自動保存到CSV文件
- 支持斷點續傳
- 可選OCR文本檢測

作者：為FYP項目創建

工作流程說明：
1. 首先設置標籤（按 t 切換文本標籤，按 0-9 選擇場景類型）
2. 然後按 'a' 接受圖像並保存到CSV
3. 或者按 'r' 拒絕圖像並移動到rejected_images文件夾（標籤會添加到文件名）
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

# 可選的OCR支持
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("提示：PaddleOCR未安裝，OCR功能不可用。運行 'pip install paddleocr' 來啟用。")

class ImageCurator:
    def __init__(self, root_folder, csv_path, use_ocr=False):
        self.root_folder = Path(root_folder)
        self.csv_path = Path(csv_path)
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        # 拒絕圖像的存放路徑
        self.rejected_folder = Path('/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images_Aerial-Traffic/rejected_images')
        # 確保拒絕文件夾存在
        self.rejected_folder.mkdir(parents=True, exist_ok=True)
        
        # 場景類型映射 - 針對Real-ESRGAN無人機圖像復原項目
        self.scene_types = {
            '0': 'other',
            '1': 'road',
            '2': 'building', 
            '3': 'infrastructure',
            '4': 'vehicle',
            '5': 'vegetation',
            '6': 'person_animal',
            '7': 'low_light',  # 低光照條件
            '8': 'motion_blur',  # 運動模糊
            '9': 'drone_blur'  # 無人機模糊或整體模糊
        }
        
        # 支持的圖像格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 初始化OCR
        if self.use_ocr:
            print("正在初始化OCR引擎...")
            self.ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
            print("OCR引擎初始化完成")
        
        # 加載或創建圖像列表
        self.image_files = self._scan_images()
        self.current_index = 0
        
        # 加載現有的CSV數據
        self.df = self._load_existing_csv()
        
        # 當前圖像的標籤狀態
        self.current_labels = {
            'has_text': False,
            'scene_types': set(),
            'accepted': False,
            'quality_level': 'high',  # 'high', 'medium', 'low'
            'is_night': False  # True=夜間, False=白天
        }
        
        print(f"找到 {len(self.image_files)} 個圖像文件")
        print(f"CSV文件路徑: {self.csv_path}")
        
    def _scan_images(self):
        """遞歸掃描圖像文件"""
        image_files = []
        for ext in self.image_extensions:
            pattern = f"**/*{ext}"
            image_files.extend(self.root_folder.glob(pattern))
            # 也搜索大寫擴展名
            pattern = f"**/*{ext.upper()}"
            image_files.extend(self.root_folder.glob(pattern))
        
        # 排序以確保一致的順序
        image_files.sort()
        return image_files
    
    def _load_existing_csv(self):
        """加載現有的CSV文件或創建新的DataFrame"""
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            print(f"加載現有CSV文件，包含 {len(df)} 條記錄")
            
            # 向後兼容：如果沒有 is_night 列，添加默認值
            if 'is_night' not in df.columns:
                df['is_night'] = 0  # 默認為白天
                print("添加夜間標籤列（默認為白天）")
            
            # 向後兼容：如果沒有 quality_level 列，添加默認值
            if 'quality_level' not in df.columns:
                df['quality_level'] = 'high'  # 默認為高質量
                print("添加質量等級列（默認為高質量）")
            
            # 找到最後處理的圖像位置以支持斷點續傳
            if len(df) > 0:
                processed_files = set(df['filepath'].values)
                for i, img_path in enumerate(self.image_files):
                    if str(img_path) not in processed_files:
                        self.current_index = i
                        print(f"從第 {i+1} 張圖像繼續處理")
                        break
                else:
                    self.current_index = len(self.image_files) - 1
                    print("所有圖像都已處理過")
        else:
            df = pd.DataFrame(columns=['filepath', 'has_text', 'scene_type', 'quality_level', 'is_night', 'timestamp'])
            print("創建新的CSV文件")
        
        return df
    
    def _detect_text_ocr(self, image_path):
        """使用OCR檢測圖像中的文本"""
        if not self.use_ocr:
            return False
        
        try:
            result = self.ocr.ocr(str(image_path), cls=True)
            if result and result[0]:
                # 檢查是否有高置信度的文本檢測
                for line in result[0]:
                    if len(line) > 1 and len(line[1]) > 1:
                        confidence = line[1][1]
                        if confidence > 0.7:  # 置信度閾值
                            return True
            return False
        except Exception as e:
            print(f"OCR檢測失敗: {e}")
            return False
    
    def _update_window_title(self):
        """更新窗口標題顯示當前狀態"""
        scene_str = ','.join(sorted(self.current_labels['scene_types'])) if self.current_labels['scene_types'] else 'none'
        # 使用ASCII字符避免顯示問題
        text_status = 'Yes' if self.current_labels['has_text'] else 'No'
        title = f"Image {self.current_index + 1}/{len(self.image_files)} | Text: {text_status} | Scene: {scene_str}"
        cv2.setWindowTitle("Image Curator", title)
    
    def _display_image(self):
        """顯示當前圖像"""
        if self.current_index >= len(self.image_files):
            return False
        
        img_path = self.image_files[self.current_index]
        
        # 重置當前標籤
        self.current_labels = {
            'has_text': False,
            'scene_types': set(),
            'accepted': False,
            'quality_level': 'high',  # 默認為高質量
            'is_night': False  # 默認為白天
        }
        
        # 如果啟用OCR，自動檢測文本
        if self.use_ocr:
            self.current_labels['has_text'] = self._detect_text_ocr(img_path)
            if self.current_labels['has_text']:
                print(f"OCR檢測到文本 (可按 't' 切換)")
        
        try:
            # 讀取並顯示圖像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"無法讀取圖像: {img_path}")
                return True
            
            # 調整圖像大小以適應屏幕
            height, width = img.shape[:2]
            max_height = 800
            if height > max_height:
                scale = max_height / height
                new_width = int(width * scale)
                img = cv2.resize(img, (new_width, max_height))
            
            cv2.imshow("Image Curator", img)
            self._update_window_title()
            
            # 安全顯示文件名，避免特殊字符問題
            try:
                safe_filename = img_path.name.encode('ascii', 'ignore').decode('ascii')
                if not safe_filename:
                    safe_filename = f"image_{self.current_index + 1}"
            except:
                safe_filename = f"image_{self.current_index + 1}"
            
            print(f"\n當前圖像: {safe_filename}")
            print(f"路徑: {img_path}")
            print(f"進度: {self.current_index + 1}/{len(self.image_files)}")
            self._print_controls()
            
            return True
            
        except Exception as e:
            print(f"顯示圖像時出錯: {e}")
            return True
    
    def _print_controls(self):
        """打印控制說明"""
        print("\n控制鍵:")
        print("  建議工作流程：先設置標籤，再按 'a' 接受或 'r' 拒絕")
        print("  a = 接受圖像並保存到CSV")
        print("  r = 拒絕圖像並移動到rejected_images文件夾（標籤會添加到文件名）")
        print("  t = 切換文本標籤 (當前: {}) [再按一次可撤銷]".format('是' if self.current_labels['has_text'] else '否'))
        print("  y = 切換夜間標籤 (當前: {}) [再按一次可撤銷]".format('夜間' if self.current_labels['is_night'] else '白天'))
        print("  q/w/e = 質量等級 (q=高質量, w=中等, e=低質量) 當前: {}".format(self.current_labels['quality_level']))
        print("  0=其他, 1=道路, 2=建築, 3=基礎設施, 4=車輛, 5=植被, 6=人/動物, 7=低光照, 8=運動模糊, 9=無人機模糊")
        print("       [再按相同數字鍵可撤銷該場景標籤]")
        print("  當前場景: {}".format(','.join(sorted(self.current_labels['scene_types'])) or '無'))
        print("  n = 下一張, b = 上一張")
        print("  s = 手動保存 (需先按 'a' 接受)")
        print("  ESC = 退出, h = 顯示幫助")
    
    def _save_current_image(self):
        """保存當前圖像的標籤到CSV"""
        if not self.current_labels['accepted']:
            print("圖像未被接受，不保存")
            return
        
        img_path = self.image_files[self.current_index]
        scene_type_str = ','.join(sorted(self.current_labels['scene_types'])) if self.current_labels['scene_types'] else 'other'
        
        # 創建新記錄
        new_record = {
            'filepath': str(img_path),
            'has_text': int(self.current_labels['has_text']),
            'scene_type': scene_type_str,
            'quality_level': self.current_labels['quality_level'],
            'is_night': int(self.current_labels['is_night']),
            'timestamp': datetime.now().isoformat()
        }
        
        # 檢查是否已存在該文件的記錄
        existing_mask = self.df['filepath'] == str(img_path)
        if existing_mask.any():
            # 更新現有記錄
            self.df.loc[existing_mask, ['has_text', 'scene_type', 'quality_level', 'is_night', 'timestamp']] = [
                new_record['has_text'], new_record['scene_type'], new_record['quality_level'], new_record['is_night'], new_record['timestamp']
            ]
            print(f"更新記錄: {img_path.name}")
        else:
            # 添加新記錄
            self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
            print(f"添加新記錄: {img_path.name}")
        
        # 立即保存到CSV
        self.df.to_csv(self.csv_path, index=False)
        print(f"已保存到 {self.csv_path}")
        print(f"標籤: 文本={'是' if self.current_labels['has_text'] else '否'}, 場景={scene_type_str}, 質量={self.current_labels['quality_level']}, 時間={'夜間' if self.current_labels['is_night'] else '白天'}")
    
    def _move_rejected_image(self):
        """移動拒絕的圖像到rejected_images文件夾並添加標籤到文件名"""
        img_path = self.image_files[self.current_index]
        
        # 構建標籤字符串
        tags = []
        if self.current_labels['has_text']:
            tags.append('text')
        if self.current_labels['scene_types']:
            tags.extend(sorted(self.current_labels['scene_types']))
        if self.current_labels['quality_level'] == 'low':
            tags.append('low-quality')
        elif self.current_labels['quality_level'] == 'medium':
            tags.append('medium-quality')
        if self.current_labels['is_night']:
            tags.append('night')
        
        # 如果沒有標籤，添加默認標籤
        if not tags:
            tags.append('rejected')
        
        # 構建新文件名
        original_stem = img_path.stem
        original_suffix = img_path.suffix
        tags_str = '_'.join(tags)
        new_filename = f"{original_stem}_{tags_str}{original_suffix}"
        
        # 目標路徑
        target_path = self.rejected_folder / new_filename
        
        # 如果目標文件已存在，添加數字後綴
        counter = 1
        while target_path.exists():
            new_filename = f"{original_stem}_{tags_str}_{counter}{original_suffix}"
            target_path = self.rejected_folder / new_filename
            counter += 1
        
        try:
            # 移動文件
            shutil.move(str(img_path), str(target_path))
            print(f"✗ 圖像已移動到: {target_path}")
            print(f"標籤: {tags_str}")
            return True
        except Exception as e:
            print(f"移動文件失敗: {e}")
            return False
    
    def run(self):
        """運行主循環"""
        print("圖像篩選工具啟動")
        print("按 'h' 查看控制說明")
        print("\n推薦工作流程：")
        print("1. 先設置標籤 (按 't' 切換文本，按 0-9 選場景)")
        print("   場景: 0=其他, 1=道路, 2=建築, 3=基礎設施, 4=車輛, 5=植被, 6=人/動物, 7=低光照, 8=運動模糊, 9=無人機模糊")
        print("2. 再按 'a' 接受並保存到CSV")
        print("3. 或按 'r' 拒絕並移動到rejected_images文件夾（標籤會添加到文件名）")
        
        cv2.namedWindow("Image Curator", cv2.WINDOW_NORMAL)
        
        if not self._display_image():
            print("沒有圖像可顯示")
            return
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC鍵
                print("退出程序")
                break
            
            elif key == ord('a'):
                self.current_labels['accepted'] = True
                print("✓ 接受圖像")
                self._save_current_image()
                self.current_index += 1
                if not self._display_image():
                    print("已處理完所有圖像")
                    break
            
            elif key == ord('r'):
                # 移動拒絕的圖像到rejected_images文件夾
                if self._move_rejected_image():
                    # 從圖像列表中移除已移動的圖像
                    self.image_files.pop(self.current_index)
                    # 調整索引，因為列表長度減少了
                    if self.current_index >= len(self.image_files):
                        self.current_index = len(self.image_files) - 1
                    
                    if not self._display_image():
                        print("已處理完所有圖像")
                        break
                else:
                    # 如果移動失敗，跳到下一張
                    print("✗ 拒絕圖像（移動失敗，跳過）")
                    self.current_index += 1
                    if not self._display_image():
                        print("已處理完所有圖像")
                        break
            
            elif key == ord('t'):
                self.current_labels['has_text'] = not self.current_labels['has_text']
                print(f"文本標籤切換為: {'是' if self.current_labels['has_text'] else '否'}")
                self._update_window_title()
            
            elif key == ord('y'):
                self.current_labels['is_night'] = not self.current_labels['is_night']
                print(f"時間標籤切換為: {'夜間' if self.current_labels['is_night'] else '白天'}")
            
            elif key == ord('q'):
                self.current_labels['quality_level'] = 'high'
                print(f"質量等級設為: 高質量 (適合作為GT)")
                
            elif key == ord('w'):
                self.current_labels['quality_level'] = 'medium'
                print(f"質量等級設為: 中等質量")
                
            elif key == ord('e'):
                self.current_labels['quality_level'] = 'low'
                print(f"質量等級設為: 低質量 (用於真實測試)")
            
            elif key in [ord(str(i)) for i in range(1, 10)]:
                scene_key = chr(key)
                scene_type = self.scene_types[scene_key]
                
                if scene_type in self.current_labels['scene_types']:
                    self.current_labels['scene_types'].remove(scene_type)
                    print(f"移除場景類型: {scene_type}")
                else:
                    self.current_labels['scene_types'].add(scene_type)
                    print(f"添加場景類型: {scene_type}")
                
                print(f"當前場景類型: {','.join(sorted(self.current_labels['scene_types'])) or '無'}")
                self._update_window_title()
            
            elif key == ord('n'):
                if self.current_index < len(self.image_files) - 1:
                    self.current_index += 1
                    self._display_image()
                    print("下一張圖像")
                else:
                    print("已經是最後一張圖像")
            
            elif key == ord('b'):
                if self.current_index > 0:
                    self.current_index -= 1
                    self._display_image()
                    print("上一張圖像")
                else:
                    print("已經是第一張圖像")
            
            elif key == ord('s'):
                if self.current_labels['accepted']:
                    self._save_current_image()
                else:
                    print("請先按 'a' 接受圖像再保存")
            
            elif key == ord('h'):
                self._print_controls()
            
            else:
                print(f"未知按鍵，按 'h' 查看幫助")
        
        cv2.destroyAllWindows()
        
        # 顯示最終統計
        print(f"\n處理完成!")
        print(f"總共接受的圖像: {len(self.df)}")
        print(f"CSV文件保存在: {self.csv_path}")
        
        if len(self.df) > 0:
            print("\n場景類型統計:")
            scene_counts = {}
            for scene_str in self.df['scene_type']:
                for scene in scene_str.split(','):
                    scene = scene.strip()
                    scene_counts[scene] = scene_counts.get(scene, 0) + 1
            
            for scene, count in sorted(scene_counts.items()):
                print(f"  {scene}: {count}")
            
            text_count = self.df['has_text'].sum()
            print(f"\n包含文本的圖像: {text_count}/{len(self.df)} ({text_count/len(self.df)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='圖像篩選和標註工具')
    parser.add_argument('--root', 
                       default='/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images_Aerial-Traffic',
                       help='圖像文件夾路徑')
    parser.add_argument('--csv', 
                       default='/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/accepted_set_Aerial-Traffic.csv',
                       help='CSV輸出文件路徑')
    parser.add_argument('--ocr', action='store_true', help='啟用OCR文本檢測')
    
    args = parser.parse_args()
    
    # 檢查路徑
    if not Path(args.root).exists():
        print(f"錯誤: 圖像文件夾不存在: {args.root}")
        return
    
    # 創建CSV文件的目錄
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    
    # 啟動工具
    curator = ImageCurator(args.root, args.csv, args.ocr)
    curator.run()

if __name__ == "__main__":
    main()
