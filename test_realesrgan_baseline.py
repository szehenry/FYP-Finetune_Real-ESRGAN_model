#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 基準測試驗證腳本
========================================

目的：快速測試流程是否正常（只處理 10 張圖像）

使用：
    python test_realesrgan_baseline.py

作者：FYP Project
日期：2025-12
"""

import sys
from pathlib import Path

# 修改配置，只測試少量圖像
def test_realesrgan_only():
    """測試 Real-ESRGAN 評估（只處理 10 張）"""
    
    print("\n" + "=" * 80)
    print("🧪 測試 Real-ESRGAN 評估流程（10 張圖像）")
    print("=" * 80)
    
    # 導入並修改配置
    try:
        from evaluate_realesrgan_only import RealESRGANOnlyConfig, RealESRGANOnlyEvaluator
    except ImportError as e:
        print(f"❌ 無法導入模組: {e}")
        print("請確保 evaluate_realesrgan_only.py 在相同目錄下")
        return False
    
    # 修改配置（測試用）
    original_sample_size = RealESRGANOnlyConfig.LPIPS_SAMPLE_SIZE
    RealESRGANOnlyConfig.LPIPS_SAMPLE_SIZE = 10  # 只測試 10 張
    
    # 修改輸出目錄
    RealESRGANOnlyConfig.OUTPUT_DIR = Path(r"D:\baseline_results_realesrgan_test")
    RealESRGANOnlyConfig.RESULTS_DIR = RealESRGANOnlyConfig.OUTPUT_DIR / "enhanced_images"
    RealESRGANOnlyConfig.LEADERBOARD_DIR = RealESRGANOnlyConfig.OUTPUT_DIR / "leaderboards"
    
    try:
        # 創建評估器
        evaluator = RealESRGANOnlyEvaluator()
        
        # 只處理前 10 張
        evaluator.pairs_df = evaluator.pairs_df.head(10)
        
        print(f"\n💡 測試模式：只處理前 10 張圖像")
        print(f"   輸出目錄: {RealESRGANOnlyConfig.OUTPUT_DIR}")
        
        # 執行評估
        evaluator.evaluate_realesrgan()
        
        print("\n✅ 測試成功！")
        print(f"   結果保存在: {RealESRGANOnlyConfig.OUTPUT_DIR}")
        print("\n💡 如果測試通過，可以運行完整版本：")
        print("   python evaluate_realesrgan_only.py")
        
        return True
    
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 恢復原配置
        RealESRGANOnlyConfig.LPIPS_SAMPLE_SIZE = original_sample_size


def test_merge():
    """測試合併功能（只處理 5 張對比圖）"""
    
    print("\n" + "=" * 80)
    print("🧪 測試結果合併流程（5 張對比圖）")
    print("=" * 80)
    
    try:
        from merge_baseline_results import MergeConfig, BaselineResultsMerger
    except ImportError as e:
        print(f"❌ 無法導入模組: {e}")
        return False
    
    # 修改配置（測試用）
    MergeConfig.NUM_SAMPLES = 5  # 只生成 5 張對比圖
    MergeConfig.OUTPUT_DIR = Path(r"D:\baseline_results_merged_test")
    MergeConfig.LEADERBOARD_DIR = MergeConfig.OUTPUT_DIR / "leaderboards"
    MergeConfig.COMPARISON_DIR = MergeConfig.OUTPUT_DIR / "comparison_samples"
    MergeConfig.FAILURE_DIR = MergeConfig.OUTPUT_DIR / "failure_cases"
    
    # 檢查必要文件是否存在
    if not MergeConfig.OLD_RESULTS_DIR.exists():
        print(f"⚠️  舊結果目錄不存在: {MergeConfig.OLD_RESULTS_DIR}")
        print("   請先運行 baseline_evaluation_with_gt-HenryCC.py")
        return False
    
    # 使用測試結果（如果存在）
    test_realesrgan_dir = Path(r"D:\baseline_results_realesrgan_test\leaderboards")
    if test_realesrgan_dir.exists():
        print(f"   使用測試 Real-ESRGAN 結果: {test_realesrgan_dir}")
        MergeConfig.REALESRGAN_RESULTS_DIR = test_realesrgan_dir
    elif not MergeConfig.REALESRGAN_RESULTS_DIR.exists():
        print(f"⚠️  Real-ESRGAN 結果不存在: {MergeConfig.REALESRGAN_RESULTS_DIR}")
        print("   請先運行 test_realesrgan_baseline.py 或 evaluate_realesrgan_only.py")
        return False
    
    try:
        # 創建合併器
        merger = BaselineResultsMerger()
        
        print(f"\n💡 測試模式：只生成 5 張對比圖")
        print(f"   輸出目錄: {MergeConfig.OUTPUT_DIR}")
        
        # 合併結果
        df_merged = merger.merge_results()
        
        if df_merged is not None:
            # 生成少量對比樣本
            merger.generate_comparison_samples(num_samples=5)
            
            print("\n✅ 測試成功！")
            print(f"   結果保存在: {MergeConfig.OUTPUT_DIR}")
            print("\n💡 如果測試通過，可以運行完整版本：")
            print("   python merge_baseline_results.py")
            
            return True
        else:
            print("\n❌ 合併失敗")
            return False
    
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    
    print("\n" + "=" * 80)
    print("🚀 Real-ESRGAN 基準測試驗證")
    print("=" * 80)
    print("\n本腳本將進行快速測試（只處理少量圖像）")
    print("確保流程正常後再運行完整版本")
    
    # 測試選項
    print("\n請選擇測試項目：")
    print("  1. 測試 Real-ESRGAN 評估（10 張圖像，約 5-10 分鐘）")
    print("  2. 測試結果合併（5 張對比圖，約 2-3 分鐘）")
    print("  3. 全部測試")
    print("  q. 退出")
    
    choice = input("\n請輸入選項 (1/2/3/q): ").strip()
    
    if choice == '1':
        test_realesrgan_only()
    
    elif choice == '2':
        test_merge()
    
    elif choice == '3':
        print("\n" + "=" * 80)
        print("📝 測試 1/2: Real-ESRGAN 評估")
        print("=" * 80)
        success1 = test_realesrgan_only()
        
        if success1:
            print("\n" + "=" * 80)
            print("📝 測試 2/2: 結果合併")
            print("=" * 80)
            success2 = test_merge()
            
            if success2:
                print("\n" + "=" * 80)
                print("✅ 所有測試通過！")
                print("=" * 80)
                print("\n您可以安全地運行完整版本：")
                print("  1. python evaluate_realesrgan_only.py")
                print("  2. python merge_baseline_results.py")
            else:
                print("\n❌ 合併測試失敗，請檢查錯誤")
        else:
            print("\n❌ Real-ESRGAN 測試失敗，請檢查錯誤")
    
    elif choice.lower() == 'q':
        print("\n退出測試")
        return
    
    else:
        print("\n❌ 無效選項")


if __name__ == "__main__":
    main()

