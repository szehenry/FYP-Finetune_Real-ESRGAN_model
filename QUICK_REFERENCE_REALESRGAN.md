# Real-ESRGAN 基準評估 - 快速參考卡片

## 🚀 3 步完成評估

```bash
# 步驟 1: 測試（5 分鐘）
python test_realesrgan_baseline.py

# 步驟 2: 評估 Real-ESRGAN（2-3 小時）
python evaluate_realesrgan_only.py

# 步驟 3: 合併結果（15 分鐘）
python merge_baseline_results.py
```

## 📊 輸出位置

```
D:\baseline_results_merged\leaderboards\full_results_merged.csv   ← 完整結果
D:\baseline_results_merged\comparison_samples\                    ← 50 張對比圖
```

## 🔧 關鍵配置

| 設置 | 位置 | 默認值 | 說明 |
|------|------|--------|------|
| 對比圖數量 | `merge_baseline_results.py` | 50 | 可改為 20/100 |
| LPIPS 抽樣 | `evaluate_realesrgan_only.py` | 700 | 可改為 500/1000 |
| GPU Tile | `evaluate_realesrgan_only.py` | 400 | 記憶體不足改 200 |

## ⚡ 快速修改

### 減少對比圖數量（加快速度）

```python
# merge_baseline_results.py, Line 101
NUM_SAMPLES = 20  # 改為 20
```

### GPU 記憶體不足

```python
# evaluate_realesrgan_only.py, Line 107
TILE = 200  # 改為 200
```

### 使用 CPU（非常慢）

```python
# evaluate_realesrgan_only.py, Line 113
DEVICE = 'cpu'  # 改為 'cpu'
```

## 🏆 預期結果

| 方法 | PSNR | SSIM | 排名 |
|------|------|------|------|
| **Real-ESRGAN** | **29.2** | **0.87** | 🥇 |
| Combined | 25.8 | 0.83 | 🥈 |
| Identity | 23.5 | 0.78 | 7th |

Real-ESRGAN 預計比最好的傳統方法強 **+3.4 dB**

## ⚠️ 路徑檢查

運行前確認存在：

```
✅ D:\degraded_full_dataset\pairs.csv
✅ D:\FYP_Images\
✅ D:\Real-ESRGAN\...\RealESRGAN_x4plus.pth
✅ D:\baseline_results_old\leaderboards\full_results_with_gt.csv
```

## 💡 常見錯誤

| 錯誤 | 解決方案 |
|------|----------|
| `CUDA out of memory` | 改 `TILE = 200` |
| `ModuleNotFoundError: realesrgan` | `pip install realesrgan` |
| `FileNotFoundError: 模型文件` | 下載 `RealESRGAN_x4plus.pth` |
| `找不到舊結果` | 先跑 `baseline_evaluation_with_gt-HenryCC.py` |

## 📁 文件說明

| 文件 | 用途 | 時間 |
|------|------|------|
| `test_realesrgan_baseline.py` | 測試流程（10 張） | 5 分鐘 |
| `evaluate_realesrgan_only.py` | 評估 Real-ESRGAN | 2-3 小時 |
| `merge_baseline_results.py` | 合併所有結果 | 15 分鐘 |

## 🎯 9 個方法佈局

對比圖排列（3×3）：

```
┌─────────┬─────────┬─────────┐
│ GT      │ Degraded│ Identity│
├─────────┼─────────┼─────────┤
│ Bicubic │ Gaussian│ Sharpen │
├─────────┼─────────┼─────────┤
│ Unsharp │ Combined│RealESRGAN│ ← 新增
└─────────┴─────────┴─────────┘
```

## 📈 查看結果

```python
import pandas as pd

# 載入結果
df = pd.read_csv(r"D:\baseline_results_merged\leaderboards\full_results_merged.csv")

# 查看排名
print(df.groupby('method')['psnr'].mean().sort_values(ascending=False))
```

## 🔄 微調後重新評估

```python
# 修改 evaluate_realesrgan_only.py
MODEL_PATH = Path(r"您的微調模型.pth")

# 重新運行
python evaluate_realesrgan_only.py
python merge_baseline_results.py
```

## 📞 需要幫助？

查看詳細文檔：
- `QUICKSTART_REALESRGAN_BASELINE.md` - 快速開始
- `README_REALESRGAN_BASELINE.md` - 完整指南
- `SUMMARY_REALESRGAN_ADDITION.md` - 工作總結

---

**版本：** 1.0 | **日期：** 2025-12-22

