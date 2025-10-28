#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV文件路徑更新腳本
用於修改accepted_set.csv文件中的filepath列
將Images/路徑更改為Images_VisDrone2019/路徑
"""

import pandas as pd
import os
from pathlib import Path

def update_csv_filepath(csv_file_path):
    """
    更新CSV文件中的filepath列
    
    Args:
        csv_file_path (str): CSV文件的完整路徑
    """
    try:
        # 讀取CSV文件
        print(f"正在讀取文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # 檢查是否存在filepath列
        if 'filepath' not in df.columns:
            print("錯誤：CSV文件中沒有找到'filepath'列")
            return False
        
        # 顯示原始數據的前幾行
        print("\n原始數據（前5行）:")
        print(df['filepath'].head())
        
        # 定義要替換的路徑部分
        old_path = '/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images/'
        new_path = '/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images_VisDrone2019/'
        
        # 計算需要更新的行數
        rows_to_update = df['filepath'].str.startswith(old_path).sum()
        print(f"\n找到 {rows_to_update} 行需要更新")
        
        # 執行路徑替換
        df['filepath'] = df['filepath'].str.replace(old_path, new_path, regex=False)
        
        # 顯示更新後的數據前幾行
        print("\n更新後的數據（前5行）:")
        print(df['filepath'].head())
        
        # 創建備份文件
        backup_path = csv_file_path.replace('.csv', '_backup.csv')
        print(f"\n正在創建備份文件: {backup_path}")
        
        # 讀取原始文件並創建備份
        original_df = pd.read_csv(csv_file_path)
        original_df.to_csv(backup_path, index=False)
        
        # 保存更新後的文件
        print(f"正在保存更新後的文件: {csv_file_path}")
        df.to_csv(csv_file_path, index=False)
        
        print(f"\n✅ 成功更新了 {rows_to_update} 行數據")
        print(f"✅ 原始文件已備份到: {backup_path}")
        print(f"✅ 更新後的文件已保存")
        
        return True
        
    except FileNotFoundError:
        print(f"錯誤：找不到文件 {csv_file_path}")
        return False
    except Exception as e:
        print(f"錯誤：處理文件時發生異常 - {str(e)}")
        return False

def main():
    """主函數"""
    # CSV文件路徑
    csv_file_path = '/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/accepted_set.csv'
    
    print("=== CSV文件路徑更新腳本 ===")
    print(f"目標文件: {csv_file_path}")
    
    # 檢查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"錯誤：文件不存在 - {csv_file_path}")
        return
    
    # 詢問用戶確認
    print("\n此腳本將會:")
    print("1. 創建原始文件的備份")
    print("2. 將所有filepath列中的'Images/'路徑更改為'Images_VisDrone2019/'")
    print("3. 保存更新後的文件")
    
    confirm = input("\n是否繼續？(y/n): ").lower().strip()
    
    if confirm in ['y', 'yes', '是']:
        success = update_csv_filepath(csv_file_path)
        if success:
            print("\n🎉 文件更新完成！")
        else:
            print("\n❌ 文件更新失敗！")
    else:
        print("操作已取消")

if __name__ == "__main__":
    main()
