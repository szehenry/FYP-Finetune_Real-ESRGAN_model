# Real-ESRGAN 環境設置指南

## 目的
為 Real-ESRGAN 基準測試創建獨立的虛擬環境，避免依賴衝突。

## 環境要求
- Python >= 3.7
- PyTorch >= 1.7
- CUDA（推薦，用於 GPU 加速）

## 設置步驟

### 1. 創建虛擬環境

```powershell
# 在 FYP 項目目錄下
cd "C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP"

# 創建新的虛擬環境
python -m venv venv_realesrgan
```

### 2. 啟動虛擬環境

```powershell
# Windows PowerShell
.\venv_realesrgan\Scripts\Activate.ps1

# 如果遇到權限問題，執行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. 安裝 PyTorch

根據您的系統選擇：

**有 CUDA (推薦)：**
```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**僅 CPU：**
```powershell
pip install torch torchvision
```

### 4. 克隆 Real-ESRGAN 倉庫

```powershell
# 在 D:\ 下克隆（輸出目錄附近）
cd D:\
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
```

### 5. 安裝依賴

```powershell
# 安裝 BasicSR
pip install basicsr

# 安裝 facexlib 和 gfpgan（用於人臉增強，可選）
pip install facexlib
pip install gfpgan

# 安裝 Real-ESRGAN 依賴
pip install -r requirements.txt

# 安裝 Real-ESRGAN
python setup.py develop
```

### 6. 驗證安裝

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import realesrgan; print('Real-ESRGAN installed successfully')"
```

## 下載預訓練模型

Real-ESRGAN 提供多個預訓練模型：

```powershell
# 進入 Real-ESRGAN 目錄
cd D:\Real-ESRGAN

# 創建模型目錄
mkdir experiments\pretrained_models

# 下載模型（選擇一個或多個）
```

**推薦模型：**
1. **RealESRGAN_x4plus.pth** - 通用圖像超分辨率（4x）
   - 下載：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

2. **RealESRNet_x4plus.pth** - 訓練時的基礎模型
   - 下載：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth

3. **RealESRGAN_x2plus.pth** - 2x 超分辨率
   - 下載：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

**下載後放置位置：**
```
D:\Real-ESRGAN\experiments\pretrained_models\
```

## 測試運行

```powershell
# 回到 FYP 項目
cd "C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP"

# 啟動環境
.\venv_realesrgan\Scripts\Activate.ps1

# 測試腳本
python realesrgan_baseline.py --test
```

## 常見問題

### 1. CUDA 版本不匹配
**症狀：** RuntimeError: CUDA error: no kernel image is available for execution

**解決：** 重新安裝對應 CUDA 版本的 PyTorch

### 2. 記憶體不足
**症狀：** CUDA out of memory

**解決：** 
- 使用 CPU 模式：在腳本中設置 `--device cpu`
- 減小批次大小
- 處理較小的圖像

### 3. 依賴衝突
**症狀：** ImportError 或版本不兼容

**解決：** 使用獨立的虛擬環境（本指南的方法）

## 環境管理

**啟動環境：**
```powershell
.\venv_realesrgan\Scripts\Activate.ps1
```

**退出環境：**
```powershell
deactivate
```

**刪除環境（如需重裝）：**
```powershell
# 先退出環境
deactivate

# 刪除目錄
Remove-Item -Recurse -Force venv_realesrgan
```

## 下一步

設置完成後，運行：
```powershell
python realesrgan_baseline.py
```

這將使用 Real-ESRGAN 處理所有退化圖像並生成評估報告。

