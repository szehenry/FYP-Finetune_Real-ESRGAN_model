# Real-ESRGAN 基準評估快速指南

## 📋 概述

本指南說明如何將**未微調的 Real-ESRGAN** 添加到基準測試中，並與傳統方法進行比較。

## 🎯 為什麼要這樣做？

- ✅ 建立 Real-ESRGAN 的**基準性能**（未微調版本）
- ✅ 與您的**微調版本**進行公平對比
- ✅ 節省時間（無需重跑所有傳統方法）
- ✅ 生成完整的 9 方法對比圖（GT + Degraded + 7 個方法）

## 🔧 環境要求

### 1. venv_baseline 環境

確保已安裝 Real-ESRGAN 相關依賴：

```bash
# 激活虛擬環境（Windows PowerShell）
.\venv_baseline\Scripts\Activate.ps1

# 安裝 Real-ESRGAN 依賴
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install basicsr
pip install facexlib
pip install gfpgan
pip install realesrgan
```

### 2. Real-ESRGAN 模型

確保模型路徑正確：
```
D:\Real-ESRGAN\experiments\pretrained_models\RealESRGAN_x4plus.pth
```

如果沒有，請下載：
```bash
# 下載預訓練模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

## 📝 運行流程

### 情況 1：已經跑過傳統方法

如果您已經運行過 `baseline_evaluation_with_gt-HenryCC.py`，結果在 `D:\baseline_results_old\leaderboards`：

```bash
# 步驟 1：只評估 Real-ESRGAN（約 2-3 小時）
python evaluate_realesrgan_only.py

# 步驟 2：合併結果並生成對比圖（約 10-15 分鐘）
python merge_baseline_results.py
```

**時間節省：** 約 4-5 小時（無需重跑傳統方法）

### 情況 2：還沒跑過任何基準測試

```bash
# 步驟 1：運行傳統方法（約 4-6 小時）
python baseline_evaluation_with_gt-HenryCC.py

# 步驟 2：評估 Real-ESRGAN（約 2-3 小時）
python evaluate_realesrgan_only.py

# 步驟 3：合併結果（約 10-15 分鐘）
python merge_baseline_results.py
```

**總時間：** 約 6-9 小時

## 📊 輸出結果

### 目錄結構

```
D:\baseline_results_merged\
├── leaderboards\
│   ├── full_results_merged.csv              # 合併後的完整結果
│   ├── leaderboard_drone_motion_blur.csv    # 各退化類型排行榜
│   ├── leaderboard_object_motion_blur.csv
│   ├── leaderboard_low_light.csv
│   └── evaluation_summary_merged.json       # 統計摘要
├── comparison_samples\
│   └── [圖像名]_comparison_full.png         # 50 張對比圖（9 個方法）
└── failure_cases\
    └── failure_cases_[方法名].csv           # 各方法的失敗案例
```

### 結果內容

#### 1. 完整排行榜（full_results_merged.csv）

包含所有方法的詳細指標：

| image_name | degradation_type | method | psnr | ssim | lpips | overall_score |
|------------|------------------|--------|------|------|-------|---------------|
| img001.jpg | Drone Motion Blur | identity | 23.45 | 0.7821 | 0.245 | 0.6234 |
| img001.jpg | Drone Motion Blur | realesrgan | **28.92** | **0.8654** | **0.123** | **0.8523** |
| ... | ... | ... | ... | ... | ... | ... |

#### 2. 對比樣本圖（50 張）

每張圖包含 **9 個子圖**，排列如下：

```
┌─────────────┬─────────────┬─────────────┐
│ Original(GT)│  Degraded   │  Identity   │
├─────────────┼─────────────┼─────────────┤
│   Bicubic   │  Gaussian   │   Sharpen   │
├─────────────┼─────────────┼─────────────┤
│   Unsharp   │  Combined   │Real-ESRGAN  │
└─────────────┴─────────────┴─────────────┘
```

## 🔍 配置說明

### evaluate_realesrgan_only.py

```python
class RealESRGANOnlyConfig:
    # 輸出目錄（避免覆蓋舊結果）
    OUTPUT_DIR = Path(r"D:\baseline_results_realesrgan_only")
    
    # LPIPS 抽樣（與 HenryCC 一致）
    LPIPS_SAMPLE_SIZE = 700  # 隨機抽樣，節省時間
    
    # Real-ESRGAN 參數（RTX 3060 優化）
    TILE = 400  # 分塊大小
    FP32 = False  # 使用 FP16 加速
    DEVICE = 'cuda'  # 自動檢測
```

### merge_baseline_results.py

```python
class MergeConfig:
    # 輸入路徑
    OLD_RESULTS_DIR = Path(r"D:\baseline_results_old\leaderboards")
    REALESRGAN_RESULTS_DIR = Path(r"D:\baseline_results_realesrgan_only\leaderboards")
    
    # 輸出路徑
    OUTPUT_DIR = Path(r"D:\baseline_results_merged")
    
    # 對比樣本數量
    NUM_SAMPLES = 50  # 增加到 50 張
    
    # 方法順序（對比圖排列）
    METHOD_ORDER = [
        'Original (GT)', 'Degraded',
        'identity', 'bicubic', 'gaussian',
        'sharpen', 'unsharp', 'combined',
        'realesrgan'
    ]
```

## 🎛️ 自定義配置

### 修改對比樣本數量

如果想要更多/更少對比圖，修改 `merge_baseline_results.py`：

```python
class MergeConfig:
    NUM_SAMPLES = 100  # 改為 100 張
```

### 修改 LPIPS 抽樣大小

如果想要更多 LPIPS 計算（更慢但更準確），修改 `evaluate_realesrgan_only.py`：

```python
class RealESRGANOnlyConfig:
    LPIPS_SAMPLE_SIZE = 1000  # 增加到 1000 張
```

### 修改 Real-ESRGAN Tile 大小

如果 GPU 記憶體不足，減小 Tile 大小：

```python
class RealESRGANOnlyConfig:
    TILE = 200  # 減小到 200（更慢但更省記憶體）
```

## ⚠️ 常見問題

### 1. GPU 記憶體不足

**現象：** `CUDA out of memory`

**解決方案：**
- 減小 TILE 大小（400 → 200）
- 或在 CPU 上運行（慢很多）：
  ```python
  DEVICE = 'cpu'
  ```

### 2. Real-ESRGAN 導入失敗

**現象：** `ModuleNotFoundError: No module named 'realesrgan'`

**解決方案：**
```bash
# 重新安裝
pip uninstall realesrgan
pip install git+https://github.com/xinntao/Real-ESRGAN.git
```

### 3. 舊結果路徑不對

**現象：** `❌ 找不到舊結果`

**解決方案：**
修改 `merge_baseline_results.py` 中的路徑：
```python
OLD_RESULTS_DIR = Path(r"您的實際路徑\leaderboards")
```

### 4. 對比圖生成很慢

**原因：** Real-ESRGAN 需要對每張圖重新推理

**解決方案：**
- 減少樣本數量（50 → 20）
- 或者跳過 Real-ESRGAN 對比圖生成（只看 CSV 結果）

## 📈 預期結果

### 性能參考（2200 張圖像）

| 方法 | PSNR (dB) | SSIM | LPIPS | 時間 |
|------|-----------|------|-------|------|
| Identity | ~23.5 | ~0.78 | ~0.25 | - |
| Bicubic | ~24.2 | ~0.80 | ~0.23 | - |
| Combined | ~25.8 | ~0.83 | ~0.19 | - |
| **Real-ESRGAN** | **~28-30** | **~0.86-0.89** | **~0.12-0.15** | 2-3h |

**註：** Real-ESRGAN 通常比傳統方法好 **3-5 dB PSNR**

## 🚀 下一步

完成基準測試後：

1. ✅ 檢查 `D:\baseline_results_merged\leaderboards\` 中的排行榜
2. ✅ 查看 `comparison_samples\` 中的對比圖
3. ✅ 開始微調 Real-ESRGAN（使用您的無人機數據集）
4. ✅ 重複評估流程，對比微調前後的性能

## 💡 提示

- **時間優化：** 先跑一小部分（10-20 張）測試流程是否正常
- **GPU 監控：** 使用 `nvidia-smi` 監控 GPU 使用情況
- **結果備份：** 定期備份結果到其他位置
- **版本記錄：** 在 `evaluation_summary_merged.json` 中記錄配置版本

## 📞 支援

如遇到問題：
1. 檢查路徑配置是否正確
2. 確認 GPU 驅動和 CUDA 版本
3. 查看錯誤日誌（保存在 `OUTPUT_DIR` 中）
4. 參考 `setup_realesrgan_env.md` 重新設置環境

---

**更新日期：** 2025-12-22
**版本：** 1.0

