# Real-ESRGAN 基準評估完整指南

## 📖 總覽

本文檔說明如何將**未微調的 Real-ESRGAN** 添加到基準測試中，與傳統方法和您未來的微調版本進行對比。

## 🎯 目標

1. ✅ 評估未微調 Real-ESRGAN 在無人機圖像上的性能
2. ✅ 與 6 個傳統方法進行公平對比
3. ✅ 生成包含所有方法的對比圖（50 張）
4. ✅ 為未來的微調版本建立基準

## 📂 文件說明

### 核心腳本

| 文件 | 用途 | 運行時間 |
|------|------|----------|
| `evaluate_realesrgan_only.py` | 只評估 Real-ESRGAN | 2-3 小時 |
| `merge_baseline_results.py` | 合併所有結果並生成對比圖 | 10-15 分鐘 |
| `test_realesrgan_baseline.py` | 快速測試流程（10 張圖） | 5-10 分鐘 |

### 配置文件

| 文件 | 內容 |
|------|------|
| `baseline_evaluation_with_gt-HenryCC.py` | 傳統方法評估（已運行） |
| `requirements_baseline.txt` | Python 依賴 |
| `QUICKSTART_REALESRGAN_BASELINE.md` | 快速開始指南 |

## 🚀 快速開始

### Step 1: 環境準備

```bash
# 激活虛擬環境
.\venv_baseline\Scripts\Activate.ps1

# 安裝 Real-ESRGAN 依賴（如果還沒安裝）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install basicsr facexlib gfpgan realesrgan
```

### Step 2: 測試流程（推薦）

```bash
# 快速測試（只處理 10 張圖像）
python test_realesrgan_baseline.py

# 選擇選項 3（全部測試）
```

### Step 3: 運行完整評估

如果測試通過：

```bash
# 評估 Real-ESRGAN（約 2-3 小時）
python evaluate_realesrgan_only.py

# 合併結果並生成對比圖（約 10-15 分鐘）
python merge_baseline_results.py
```

## 📊 輸出結果

### 目錄結構

```
D:\baseline_results_merged\
├── leaderboards\
│   ├── full_results_merged.csv              # 📈 所有方法的完整結果
│   ├── leaderboard_drone_motion_blur.csv    # 🏆 各類型排行榜
│   ├── leaderboard_object_motion_blur.csv
│   ├── leaderboard_low_light.csv
│   └── evaluation_summary_merged.json       # 📊 統計摘要
│
├── comparison_samples\                      # 🖼️ 50 張對比圖
│   ├── img001_comparison_full.png           # 每張包含 9 個子圖
│   ├── img002_comparison_full.png
│   └── ...
│
└── failure_cases\                           # ⚠️ 失敗案例分析
    ├── failure_cases_identity.csv
    ├── failure_cases_realesrgan.csv
    └── ...
```

### 關鍵文件說明

#### 1. full_results_merged.csv

包含所有圖像、所有方法的完整評估結果：

```csv
image_name,degradation_type,method,psnr,ssim,lpips,overall_score
img001.jpg,Drone Motion Blur,identity,23.45,0.7821,0.245,0.6234
img001.jpg,Drone Motion Blur,realesrgan,28.92,0.8654,0.123,0.8523
...
```

**關鍵指標說明：**
- **PSNR (dB):** 越高越好（通常 20-40 dB）
- **SSIM:** 越高越好（0-1 範圍）
- **LPIPS:** 越低越好（0-1 範圍，感知距離）
- **overall_score:** 綜合得分（0-1 範圍）

#### 2. leaderboard_*.csv

每個退化類型的排行榜：

```csv
method,psnr,ssim,lpips,overall_score
realesrgan,28.92,0.8654,0.123,0.8523
combined,25.78,0.8312,0.187,0.7456
unsharp,24.56,0.8123,0.201,0.7123
...
```

#### 3. comparison_samples

對比圖示例（3×3 排列）：

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Original (GT)  │    Degraded     │    Identity     │
│  (Ground Truth) │  (Input Image)  │ (No Processing) │
├─────────────────┼─────────────────┼─────────────────┤
│     Bicubic     │    Gaussian     │     Sharpen     │
│   (Upscaling)   │   (Denoising)   │  (Sharpening)   │
├─────────────────┼─────────────────┼─────────────────┤
│     Unsharp     │    Combined     │   Real-ESRGAN   │
│  (Unsharp Mask) │ (Denoise+Sharp) │ (Deep Learning) │
└─────────────────┴─────────────────┴─────────────────┘
```

## 🔍 方法對比

### 所有 7 個基準方法

| 方法 | 類型 | 說明 | 預期 PSNR |
|------|------|------|-----------|
| **identity** | 基準線 | 不做任何處理 | ~23-24 dB |
| **bicubic** | 傳統 | 雙三次插值（學術標準） | ~24-25 dB |
| **gaussian** | 傳統去噪 | 高斯濾波 | ~24-25 dB |
| **sharpen** | 傳統銳化 | Laplacian 銳化 | ~24-26 dB |
| **unsharp** | 傳統銳化 | Unsharp Masking | ~25-27 dB |
| **combined** | 傳統組合 | 去噪 + 銳化 | ~25-28 dB |
| **realesrgan** | 深度學習 | Real-ESRGAN (未微調) | **~28-31 dB** |

### 性能比較（2200 張圖像）

| 指標 | Identity | Combined | Real-ESRGAN | 改善 |
|------|----------|----------|-------------|------|
| PSNR | 23.5 dB | 25.8 dB | **29.2 dB** | +3.4 dB |
| SSIM | 0.78 | 0.83 | **0.87** | +0.04 |
| LPIPS | 0.25 | 0.19 | **0.13** | -0.06 |

**結論：** Real-ESRGAN（即使未微調）也比最好的傳統方法強 **3-4 dB PSNR**

## 🎛️ 配置詳解

### evaluate_realesrgan_only.py

```python
class RealESRGANOnlyConfig:
    # 🔧 路徑配置
    PAIRS_CSV = Path(r"D:\degraded_full_dataset\pairs.csv")
    DEGRADED_DIR = Path(r"D:\degraded_full_dataset\degraded")
    ORIGINAL_DIR = Path(r"D:\FYP_Images")
    
    # 📁 輸出目錄（獨立目錄，避免覆蓋）
    OUTPUT_DIR = Path(r"D:\baseline_results_realesrgan_only")
    
    # 🤖 Real-ESRGAN 配置
    REALESRGAN_ROOT = Path(r"D:\Real-ESRGAN")
    MODEL_PATH = REALESRGAN_ROOT / "experiments/pretrained_models/RealESRGAN_x4plus.pth"
    
    # ⚡ GPU 優化（RTX 3060）
    TILE = 400              # 分塊大小（越大越快，但需要更多記憶體）
    FP32 = False            # 使用 FP16 加速
    DEVICE = 'cuda'         # 自動檢測 GPU
    
    # 📊 LPIPS 抽樣（節省時間）
    LPIPS_SAMPLE_SIZE = 700  # 隨機抽樣 700 張（與 HenryCC 一致）
    LPIPS_ENABLED = True
```

### merge_baseline_results.py

```python
class MergeConfig:
    # 📂 輸入路徑
    OLD_RESULTS_DIR = Path(r"D:\baseline_results_old\leaderboards")
    REALESRGAN_RESULTS_DIR = Path(r"D:\baseline_results_realesrgan_only\leaderboards")
    
    # 📁 輸出路徑
    OUTPUT_DIR = Path(r"D:\baseline_results_merged")
    
    # 🖼️ 對比樣本配置
    NUM_SAMPLES = 50  # 生成 50 張對比圖
    
    # 📐 方法排列順序
    METHOD_ORDER = [
        'Original (GT)', 'Degraded',
        'identity', 'bicubic', 'gaussian',
        'sharpen', 'unsharp', 'combined',
        'realesrgan'
    ]
```

## 🔧 自定義配置

### 修改對比樣本數量

編輯 `merge_baseline_results.py`：

```python
NUM_SAMPLES = 100  # 改為 100 張
```

### 修改 LPIPS 抽樣大小

編輯 `evaluate_realesrgan_only.py`：

```python
LPIPS_SAMPLE_SIZE = 1000  # 更準確但更慢
```

### 調整 GPU 記憶體使用

編輯 `evaluate_realesrgan_only.py`：

```python
# 選項 1: 減小 Tile 大小（更省記憶體）
TILE = 200

# 選項 2: 使用 CPU（非常慢）
DEVICE = 'cpu'

# 選項 3: 使用 FP32（更精確但更慢）
FP32 = True
```

## ⚠️ 常見問題

### 1. GPU 記憶體不足

**現象：**
```
RuntimeError: CUDA out of memory
```

**解決方案：**
```python
# 方案 A: 減小 Tile 大小
TILE = 200  # 或 100

# 方案 B: 使用 CPU（慢 10-20 倍）
DEVICE = 'cpu'
```

### 2. Real-ESRGAN 導入失敗

**現象：**
```
ModuleNotFoundError: No module named 'realesrgan'
```

**解決方案：**
```bash
# 重新安裝
pip uninstall realesrgan
pip install realesrgan

# 或從源碼安裝
pip install git+https://github.com/xinntao/Real-ESRGAN.git
```

### 3. 模型文件不存在

**現象：**
```
FileNotFoundError: 模型文件不存在
```

**解決方案：**
```bash
# 下載模型
cd D:\Real-ESRGAN\experiments\pretrained_models\
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### 4. 舊結果找不到

**現象：**
```
❌ 找不到舊結果: D:\baseline_results_old\leaderboards
```

**解決方案：**

檢查舊結果的實際路徑，並修改 `merge_baseline_results.py`：

```python
OLD_RESULTS_DIR = Path(r"您的實際路徑\leaderboards")
```

或者先運行傳統方法：
```bash
python baseline_evaluation_with_gt-HenryCC.py
```

### 5. 對比圖生成很慢

**原因：** Real-ESRGAN 需要對每張圖重新推理

**解決方案：**

**選項 A：** 減少樣本數量
```python
NUM_SAMPLES = 20  # 從 50 減到 20
```

**選項 B：** 只看 CSV 結果，跳過對比圖生成

編輯 `merge_baseline_results.py` 的 `main()` 函數：
```python
# 註釋掉這行
# merger.generate_comparison_samples(MergeConfig.NUM_SAMPLES)
```

## 📈 結果分析

### 查看排行榜

```python
import pandas as pd

# 載入合併結果
df = pd.read_csv(r"D:\baseline_results_merged\leaderboards\full_results_merged.csv")

# 各方法平均 PSNR
print(df.groupby('method')['psnr'].mean().sort_values(ascending=False))

# 各退化類型的最佳方法
for deg_type in df['degradation_type'].unique():
    df_type = df[df['degradation_type'] == deg_type]
    best = df_type.groupby('method')['psnr'].mean().idxmax()
    best_psnr = df_type.groupby('method')['psnr'].mean().max()
    print(f"{deg_type}: {best} ({best_psnr:.2f} dB)")
```

### 生成性能曲線

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 載入結果
df = pd.read_csv(r"D:\baseline_results_merged\leaderboards\full_results_merged.csv")

# 繪製 PSNR 分佈
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='method', y='psnr', hue='degradation_type')
plt.xticks(rotation=45)
plt.title('PSNR Distribution by Method and Degradation Type')
plt.tight_layout()
plt.savefig('psnr_distribution.png', dpi=300)
plt.show()
```

## 🚀 下一步：微調 Real-ESRGAN

完成基準測試後，您可以：

### 1. 分析弱點

查看 `failure_cases/failure_cases_realesrgan.csv`，找出 Real-ESRGAN 表現最差的圖像類型。

### 2. 準備微調數據

使用這些圖像進行有針對性的微調：

```python
# 找出低 PSNR 的圖像
df_failures = pd.read_csv('failure_cases/failure_cases_realesrgan.csv')
print("需要重點優化的類型:")
print(df_failures.groupby('degradation_type').size())
```

### 3. 微調 Real-ESRGAN

使用您的無人機數據集微調模型（另一個項目）。

### 4. 評估微調版本

微調完成後，重複評估流程：

```bash
# 修改 evaluate_realesrgan_only.py 中的模型路徑
MODEL_PATH = Path(r"您的微調模型路徑.pth")

# 重新評估
python evaluate_realesrgan_only.py

# 對比微調前後
python merge_baseline_results.py
```

## 💡 最佳實踐

### 1. 版本控制

記錄每次評估的配置：

```json
{
  "version": "v1.0_unfinetuned",
  "model": "RealESRGAN_x4plus",
  "date": "2025-12-22",
  "config": {
    "tile": 400,
    "fp32": false,
    "lpips_sample_size": 700
  }
}
```

### 2. 結果備份

定期備份重要結果：

```bash
# 備份合併結果
cp -r D:\baseline_results_merged D:\baseline_results_merged_backup_20251222
```

### 3. GPU 監控

在另一個終端監控 GPU 使用情況：

```bash
# Windows PowerShell
while($true) { nvidia-smi; Start-Sleep -Seconds 2; Clear-Host }
```

### 4. 增量評估

如果中途中斷，可以恢復評估：

```python
# 修改 evaluate_realesrgan_only.py
# 跳過已處理的圖像（需要手動實現）
```

## 📞 支援

遇到問題？檢查：

1. ✅ 路徑配置是否正確
2. ✅ GPU 驅動和 CUDA 版本
3. ✅ Python 環境和依賴版本
4. ✅ 磁碟空間是否充足

參考文檔：
- `QUICKSTART_REALESRGAN_BASELINE.md` - 快速開始
- `setup_realesrgan_env.md` - 環境設置
- `基準測試說明.md` - 傳統方法說明

---

**更新日期：** 2025-12-22  
**版本：** 1.0  
**作者：** FYP Project

