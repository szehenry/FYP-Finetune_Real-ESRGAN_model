"""
下載 OneDrive 影像到本地
確保所有需要處理的影像都在本地可用
"""

import os
import time
from pathlib import Path
from tqdm import tqdm

def read_image_list(list_file):
    """讀取影像列表"""
    with open(list_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def ensure_file_available(file_path, timeout=30):
    """
    確保檔案在本地可用
    嘗試讀取檔案以觸發 OneDrive 下載
    """
    try:
        if not os.path.exists(file_path):
            return False, "檔案不存在"
        
        # 嘗試打開檔案（這會觸發 OneDrive 下載）
        start_time = time.time()
        with open(file_path, 'rb') as f:
            # 讀取前幾個位元組以觸發下載
            f.read(1024)
        
        # 檢查檔案大小
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # 如果檔案太小，可能還沒完全下載
            time.sleep(1)
            file_size = os.path.getsize(file_path)
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            return False, f"下載超時 ({elapsed:.1f}s)"
        
        return True, f"已就緒 ({file_size/1024:.1f} KB)"
    
    except Exception as e:
        return False, str(e)

def main():
    # 讀取所有影像列表
    split_dir = Path("own_Windows_data_split_results")
    
    all_images = []
    for split_name in ['train', 'val', 'test']:
        list_file = split_dir / f'{split_name}_list.txt'
        if list_file.exists():
            images = read_image_list(list_file)
            all_images.extend(images)
            print(f"✓ {split_name}: {len(images)} 張影像")
    
    print(f"\n總共需要檢查 {len(all_images)} 張影像\n")
    
    # 檢查並下載
    unavailable = []
    failed = []
    
    with tqdm(all_images, desc="檢查影像可用性") as pbar:
        for img_path in pbar:
            success, message = ensure_file_available(img_path, timeout=30)
            
            if not success:
                if "檔案不存在" in message:
                    unavailable.append((img_path, message))
                else:
                    failed.append((img_path, message))
                pbar.set_postfix_str(f"失敗: {len(unavailable) + len(failed)}")
    
    # 報告結果
    print("\n" + "="*80)
    print("檢查完成！")
    print("="*80)
    print(f"✓ 成功: {len(all_images) - len(unavailable) - len(failed)} 張")
    print(f"✗ 檔案不存在: {len(unavailable)} 張")
    print(f"✗ 下載失敗: {len(failed)} 張")
    
    if unavailable:
        print("\n檔案不存在的影像:")
        for img_path, msg in unavailable[:10]:  # 只顯示前 10 個
            print(f"  - {img_path}")
        if len(unavailable) > 10:
            print(f"  ... 還有 {len(unavailable) - 10} 個")
    
    if failed:
        print("\n下載失敗的影像:")
        for img_path, msg in failed[:10]:
            print(f"  - {img_path}: {msg}")
        if len(failed) > 10:
            print(f"  ... 還有 {len(failed) - 10} 個")
    
    # 儲存問題清單
    if unavailable or failed:
        with open("problematic_images.txt", "w", encoding="utf-8") as f:
            f.write("檔案不存在:\n")
            for img_path, msg in unavailable:
                f.write(f"{img_path}\n")
            f.write("\n下載失敗:\n")
            for img_path, msg in failed:
                f.write(f"{img_path}: {msg}\n")
        print(f"\n✓ 問題清單已儲存到: problematic_images.txt")

if __name__ == '__main__':
    main()

