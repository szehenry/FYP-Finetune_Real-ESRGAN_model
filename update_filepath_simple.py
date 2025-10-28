#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單的CSV文件路徑更新腳本
自動修改accepted_set.csv文件中的filepath列
"""

import pandas as pd
import os

def update_filepath():
    """更新CSV文件中的filepath列"""
    
    # CSV文件路徑
    csv_file = '/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/accepted_set.csv'
    
    try:
        # 讀取CSV文件
        print("正在讀取CSV文件...")
        df = pd.read_csv(csv_file)
        
        # 顯示原始路徑示例
        print(f"原始路徑示例: {df['filepath'].iloc[0]}")
        
        # 定義路徑替換
        old_path = '/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images/'
        new_path = '/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images_VisDrone2019/'
        
        # 計算需要更新的行數
        rows_to_update = df['filepath'].str.startswith(old_path).sum()
        print(f"需要更新的行數: {rows_to_update}")
        
        # 創建備份
        backup_file = csv_file.replace('.csv', '_backup.csv')
        df.to_csv(backup_file, index=False)
        print(f"已創建備份文件: {backup_file}")
        
        # 執行路徑替換
        df['filepath'] = df['filepath'].str.replace(old_path, new_path, regex=False)
        
        # 保存更新後的文件
        df.to_csv(csv_file, index=False)
        
        # 顯示更新後的路徑示例
        print(f"更新後路徑示例: {df['filepath'].iloc[0]}")
        print(f"✅ 成功更新了 {rows_to_update} 行數據")
        
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")

if __name__ == "__main__":
    update_filepath()
