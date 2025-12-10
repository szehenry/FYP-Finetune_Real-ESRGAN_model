# 基準測試快速開始指南
# Baseline Testing Quick Start Guide

**5分鐘開始基準測試！**

---

## 🚀 最快開始方式

### 步驟 1：創建環境

```powershell
# 在 FYP 目錄
cd "C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP"

# 創建虛擬環境
python -m venv venv_baseline

# 啟動環境
.\venv_baseline\Scripts\Activate.ps1
```

### 步驟 2：安裝依賴

```powershell
# 如果有 CUDA GPU（推薦）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 如果只有 CPU
pip install torch torchvision

# 安裝其他依賴
pip install -r requirements_baseline.txt
```

### 步驟 3：運行基準測試

```powershell
# 運行簡單基準測試
python baseline_evaluation.py
```

就這樣！等待處理完成，結果將保存在 `D:\baseline_results\`

---

## 📊 查看結果

### 1. 打開排行榜

```powershell
# 在 Windows 檔案總管中打開
explorer D:\baseline_results\leaderboards

# 或用 Excel/記事本打開 CSV
start D:\baseline_results\leaderboards\full_results.csv
```

### 2. 查看對比圖像

```powershell
# 打開對比樣本目錄
explorer D:\baseline_results\comparison_samples
```

### 3. 關鍵檔案

- **full_results.csv**：所有圖像的詳細指標
- **leaderboard_drone_motion_blur.csv**：無人機運動模糊排行榜
- **leaderboard_object_motion_blur.csv**：物體運動模糊排行榜
- **leaderboard_low_light.csv**：低光照排行榜

---

## 🎯 進階：加入 Real-ESRGAN

### 需要額外時間：約 15-20 分鐘設置

### 步驟 1：創建 Real-ESRGAN 環境

```powershell
# 創建新環境
python -m venv venv_realesrgan

# 啟動
.\venv_realesrgan\Scripts\Activate.ps1

# 安裝 PyTorch（根據您的 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步驟 2：安裝 Real-ESRGAN

```powershell
# 克隆倉庫到 D:\
cd D:\
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# 安裝依賴
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
```

### 步驟 3：下載模型

**手動下載：**
1. 訪問：https://github.com/xinntao/Real-ESRGAN/releases
2. 下載：`RealESRGAN_x4plus.pth`（約 65 MB）
3. 放置到：`D:\Real-ESRGAN\experiments\pretrained_models\`

**或使用 PowerShell 下載：**

```powershell
# 創建目錄
mkdir D:\Real-ESRGAN\experiments\pretrained_models

# 下載模型（需要 curl 或 wget）
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "D:\Real-ESRGAN\experiments\pretrained_models\RealESRGAN_x4plus.pth"
```

### 步驟 4：測試運行

```powershell
# 回到 FYP 目錄
cd "C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP"

# 啟動 Real-ESRGAN 環境
.\venv_realesrgan\Scripts\Activate.ps1

# 測試（僅處理 5 張圖）
python realesrgan_baseline.py --test

# 如果測試成功，運行完整測試
python realesrgan_baseline.py
```

結果保存在 `D:\realesrgan_results\`

---

## 🔧 常見問題快速解決

### ❌ 問題：PowerShell 執行策略錯誤

```
.\venv_baseline\Scripts\Activate.ps1 : 無法載入...
```

**解決：**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ 問題：CUDA 不可用

```
torch.cuda.is_available() = False
```

**解決方案 1：** 安裝對應 CUDA 版本的 PyTorch

檢查 CUDA 版本：
```powershell
nvidia-smi
```

安裝對應版本：
- CUDA 11.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

**解決方案 2：** 使用 CPU（較慢但可用）
```powershell
pip install torch torchvision
```

### ❌ 問題：記憶體不足 (Out of Memory)

```
CUDA out of memory
```

**解決：**
1. 關閉其他程式釋放記憶體
2. 修改 `baseline_evaluation.py` 中的批次大小
3. 使用 CPU 模式：
   ```python
   # 在 Config 類中修改
   DEVICE = 'cpu'
   ```

### ❌ 問題：找不到退化圖像

```
❌ 沒有找到退化圖像
```

**檢查：**
```powershell
# 確認目錄存在
ls D:\degraded_full_dataset

# 確認子目錄存在
ls D:\degraded_full_dataset\drone_motion_blur
ls D:\degraded_full_dataset\object_motion_blur
ls D:\degraded_full_dataset\low_light
```

如果不存在，請先運行退化合成：
```powershell
python degradation_synthesis_parallel.py
```

### ❌ 問題：LPIPS 或 NIQE 無法安裝

```
ERROR: Could not find a version that satisfies...
```

**解決：**
這些是可選依賴，可以跳過：

```powershell
# 安裝核心依賴
pip install numpy opencv-python pandas tqdm matplotlib scikit-image scipy

# 跳過 lpips 和 pyiqa
# 腳本會自動跳過這些指標
```

---

## 📈 預期運行時間

根據您的硬件：

### 簡單基準測試（baseline_evaluation.py）

| 硬件配置 | 處理時間（假設 3000 張圖） |
|---------|------------------------|
| CPU only | 30-60 分鐘 |
| CPU + GPU (CUDA) | 10-20 分鐘 |
| 高階 GPU (RTX 3080+) | 5-10 分鐘 |

### Real-ESRGAN 基準測試

| 硬件配置 | 處理時間（假設 3000 張圖） |
|---------|------------------------|
| CPU only | 2-4 小時 ⚠️ |
| GPU (GTX 1060) | 40-60 分鐘 |
| GPU (RTX 3080) | 15-25 分鐘 |
| GPU (RTX 4090) | 8-12 分鐘 |

**建議：** 如果使用 CPU，先用 `--test` 模式測試少量圖像

---

## 📁 輸出檔案說明

### D:\baseline_results\

```
baseline_results\
├── leaderboards\                    # ⭐ 最重要：排行榜
│   ├── full_results.csv             # 所有結果
│   ├── leaderboard_drone_motion_blur.csv
│   ├── leaderboard_object_motion_blur.csv
│   └── leaderboard_low_light.csv
├── comparison_samples\              # 🖼️ 視覺對比
│   ├── image1_comparison.png        # 並排顯示所有方法
│   ├── image2_comparison.png
│   └── ...
├── enhanced_images\                 # 增強後的圖像
│   ├── drone_motion_blur\
│   ├── object_motion_blur\
│   └── low_light\
└── failure_cases\                   # 🔍 失敗案例
    ├── failure_cases_identity.csv
    ├── failure_cases_bicubic.csv
    └── ...
```

### D:\realesrgan_results\

```
realesrgan_results\
├── enhanced\                        # Real-ESRGAN 增強圖像
│   ├── drone_motion_blur\
│   ├── object_motion_blur\
│   └── low_light\
├── comparisons\                     # 前後對比
│   ├── image1_realesrgan.png
│   └── ...
└── leaderboards\
    └── realesrgan_results.csv       # ⭐ Real-ESRGAN 指標
```

---

## 🎓 結果解讀速查

### 查看哪個方法最好？

打開任一排行榜 CSV，看 **overall_score** 列：

| 得分範圍 | 評價 |
|---------|-----|
| 0.90 - 1.00 | 🏆 優秀 |
| 0.80 - 0.90 | ✅ 良好 |
| 0.70 - 0.80 | ⚠️ 中等 |
| < 0.70 | ❌ 需改進 |

### 各指標越高越好 ⬆️

- PSNR：越高越好（> 30 dB 為良好）
- SSIM：越高越好（> 0.8 為良好）
- Var Laplacian：越高越清晰
- Tenengrad：越高越清晰
- Entropy：越高細節越多
- Contrast：越高對比度越好

### 各指標越低越好 ⬇️

- LPIPS：越低越相似（< 0.2 為良好）
- NIQE：越低品質越好（< 5 為良好）
- BRISQUE：越低品質越好（< 40 為良好）
- Blur Extent：越低越清晰（< 0.4 為清晰）

---

## 🔄 完整工作流程總結

### 階段 1：簡單基準（今天完成）

```powershell
# 1. 設置環境
python -m venv venv_baseline
.\venv_baseline\Scripts\Activate.ps1
pip install -r requirements_baseline.txt

# 2. 運行測試
python baseline_evaluation.py

# 3. 查看結果
explorer D:\baseline_results\leaderboards
```

### 階段 2：Real-ESRGAN（可選，推薦）

```powershell
# 1. 設置環境
python -m venv venv_realesrgan
.\venv_realesrgan\Scripts\Activate.ps1
# ... 安裝 Real-ESRGAN（見上方）

# 2. 運行測試
python realesrgan_baseline.py --test  # 先測試
python realesrgan_baseline.py         # 完整運行

# 3. 查看結果
explorer D:\realesrgan_results\leaderboards
```

### 階段 3：分析結果

1. **比較排行榜**：哪種方法最好？
2. **查看對比圖**：視覺效果如何？
3. **分析失敗案例**：哪些圖像最困難？
4. **設定目標**：新模型要達到什麼性能？

### 階段 4：撰寫報告

在 FYP 論文中：
- 展示排行榜表格
- 包含對比圖像
- 討論各方法的優缺點
- 說明為何需要深度學習模型

---

## ✅ 檢查清單

完成以下檢查確認一切正常：

- [ ] ✅ 虛擬環境創建成功
- [ ] ✅ 依賴安裝完成（無錯誤）
- [ ] ✅ PyTorch 可以使用 CUDA（或確認使用 CPU）
- [ ] ✅ 找到退化圖像（3個子目錄都有圖像）
- [ ] ✅ 基準測試運行成功
- [ ] ✅ 生成了排行榜 CSV 檔案
- [ ] ✅ 生成了對比圖像
- [ ] ✅ 能夠打開並查看結果
- [ ] （可選）✅ Real-ESRGAN 環境設置成功
- [ ] （可選）✅ 下載了預訓練模型
- [ ] （可選）✅ Real-ESRGAN 測試運行成功

---

## 📞 需要幫助？

### 檢查日誌

如果出現錯誤，查看終端輸出的詳細信息。

### 詳細文檔

- **完整指標說明**：`基準測試說明.md`
- **Real-ESRGAN 詳細設置**：`setup_realesrgan_env.md`

### 常見錯誤排查

1. **import 錯誤**：確認虛擬環境已啟動
2. **路徑錯誤**：確認 `D:\degraded_full_dataset` 存在
3. **CUDA 錯誤**：使用 CPU 模式或重新安裝 PyTorch
4. **記憶體錯誤**：減少批次大小或使用 `--test` 模式

---

## 🎉 完成後的下一步

恭喜！您已經完成了基準測試階段。

**現在您有了：**
✅ 多種基準方法的性能數據  
✅ 詳細的評估指標  
✅ 視覺化對比  
✅ 失敗案例分析  

**下一階段（Week 7-10）：**
- 設計並訓練深度學習模型
- 使用這些基準進行比較
- 在論文中展示改進效果

**預期目標：**
- 新模型的 PSNR 應比最佳基準高 2-5 dB
- SSIM 應高 0.05-0.1
- 視覺效果明顯更好

祝您的 FYP 順利！ 🚀

