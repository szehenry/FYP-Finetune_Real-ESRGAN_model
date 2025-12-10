# Windows 系統降質合成工具安裝指南
# Installation Guide for Degradation Synthesis on Windows

## 系統需求確認 (System Requirements Check)

### 1. 檢查 Windows 版本
- 按下 `Win + R`，輸入 `winver`
- 需要 **Windows 10** 或 **Windows 11**

### 2. 檢查 Python 版本
開啟 **命令提示字元** (Command Prompt) 或 **PowerShell**：

```cmd
python --version
```

應該顯示 `Python 3.8` 或更高版本。

**如果沒有安裝 Python**：
1. 前往 https://www.python.org/downloads/
2. 下載 Python 3.10 或 3.11（推薦）
3. 安裝時**務必勾選** "Add Python to PATH"

### 3. 檢查 NVIDIA 驅動程式和 CUDA

因為你有 **RTX 4060**，我們要確保 CUDA 正確安裝。

```cmd
nvidia-smi
```

應該會顯示類似這樣：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
```

**如果沒有顯示或出現錯誤**：
1. 前往 NVIDIA 官網下載最新驅動程式
2. 網址: https://www.nvidia.com/download/index.aspx
3. 選擇：
   - Product Type: GeForce
   - Product Series: GeForce RTX 40 Series
   - Product: GeForce RTX 4060
   - Operating System: Windows 11 或 Windows 10

## 安裝步驟 (Step-by-Step Installation)

### 步驟 1: 開啟專案資料夾

1. 開啟 **檔案總管**
2. 導航到你的專案資料夾：
   ```
   C:\Users\你的使用者名稱\OneDrive-TheHongKongPolytechnicUniversity\Y4_SEM1\FYP
   ```
3. 在資料夾空白處按住 `Shift` + 右鍵，選擇「在此處開啟 PowerShell 視窗」

   如果只有「命令提示字元」選項也可以。

### 步驟 2: 創建虛擬環境

在 PowerShell 或命令提示字元中執行：

```powershell
# 創建虛擬環境
python -m venv venv_degradation

# 啟動虛擬環境
.\venv_degradation\Scripts\activate
```

**如果出現「無法載入，因為這個系統已停用指令碼執行」錯誤**：

這是 Windows PowerShell 的安全性限制。解決方法：

**方法 1（推薦）**：使用命令提示字元 (CMD) 而非 PowerShell
```cmd
venv_degradation\Scripts\activate.bat
```

**方法 2**：暫時允許 PowerShell 執行腳本
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv_degradation\Scripts\activate
```

成功後，你的命令提示符前面會出現 `(venv_degradation)`，表示虛擬環境已啟動。

### 步驟 3: 升級 pip

```cmd
python -m pip install --upgrade pip
```

### 步驟 4: 安裝 PyTorch (GPU 版本)

這是**最重要**的步驟！RTX 4060 需要支援 CUDA 的 PyTorch。

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**說明**：
- `cu118` 表示 CUDA 11.8
- 這個版本相容於大多數 RTX 系列顯卡
- 下載大小約 2-3 GB，需要一些時間

**驗證安裝**：
```cmd
python -c "import torch; print('PyTorch 版本:', torch.__version__)"
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
python -c "import torch; print('GPU 名稱:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**預期輸出**：
```
PyTorch 版本: 2.x.x+cu118
CUDA 可用: True
GPU 名稱: NVIDIA GeForce RTX 4060
```

**如果 CUDA 可用顯示 False**：
- 重新安裝 NVIDIA 驅動程式
- 確認使用正確的 PyTorch CUDA 版本
- 重新啟動電腦

### 步驟 5: 安裝其他依賴套件

```cmd
pip install -r requirements_degradation.txt
```

這會安裝：
- numpy
- opencv-python
- ultralytics (YOLOv8)
- tqdm
- Pillow
- scipy

### 步驟 6: 驗證 YOLOv8 安裝

```cmd
python -c "from ultralytics import YOLO; print('YOLOv8 安裝成功')"
```

第一次執行時，YOLOv8 會自動下載預訓練模型（約 6 MB）。

## 測試安裝 (Test Installation)

### 快速測試單張影像

```cmd
# 從你的資料集中選一張影像測試
python test_degradation.py --image "C:\path\to\your\image.jpg" --output .\test_output --show
```

**範例**（使用你實際的影像路徑）：
```cmd
python test_degradation.py --image "C:\Users\YourName\OneDrive-TheHongKongPolytechnicUniversity\Y4_SEM1\FYP_Images\Images_Aerial-Traffic\img_1.jpg" --output .\test_output
```

這個測試會：
1. 生成 3 種降質效果（全局模糊、物體模糊、低光照）
2. 儲存到 `test_output` 資料夾
3. 創建對比圖 `comparison_grid.jpg`

**檢查輸出**：
開啟 `test_output\comparison_grid.jpg` 查看效果。

### 測試完整流程（小規模）

```cmd
# 只處理每個分割的前 5 張影像
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\test_degraded_small --max_images 5
```

預期輸出：
- Train: 5 張 × 3 種降質 = 15 對
- Val: 5 張 × 3 種降質 = 15 對
- Test: 5 張 × 3 種降質 = 15 對
- **總計**: 約 45 對降質影像

處理時間（RTX 4060）：約 30 秒 - 1 分鐘

## 開始正式處理 (Start Full Processing)

確認測試成功後，可以處理完整資料集：

```cmd
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data
```

根據你的資料集統計：
- Train: 3069 張
- Val: 636 張
- Test: 686 張
- **總計**: 4391 張原始影像

生成降質影像（假設每張生成 2-3 種降質）：
- **預計**: 約 10,000 - 13,000 對降質影像
- **處理時間（RTX 4060）**: 約 2-3 小時
- **硬碟空間**: 約 5-10 GB（依影像大小而定）

**建議**：
1. 確保電腦插著電源（不要用電池）
2. 關閉省電模式
3. 不要讓電腦進入睡眠狀態
4. 可以在背景執行，但避免同時進行其他高負載任務

## 監控處理進度 (Monitor Progress)

### 查看 GPU 使用情況

開啟另一個 PowerShell/CMD 視窗：

```cmd
# 持續監控 GPU
nvidia-smi -l 1
```

按 `Ctrl+C` 停止監控。

**正常狀態**：
- GPU-Util: 30-80%（物體模糊時較高）
- Memory-Usage: 1-3 GB / 8 GB
- Power: 50-120W

### 查看輸出檔案

處理過程中，你可以隨時查看 `degraded_data` 資料夾：

```
degraded_data\
├── degraded\           # 降質影像
├── metadata\           # 參數記錄
├── pairs.csv          # 配對清單（處理完才會生成）
└── degradation_summary.json  # 摘要（處理完才會生成）
```

## 常見問題排解 (Troubleshooting)

### 問題 1: "CUDA out of memory"

**錯誤訊息**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解決方法**：
1. 關閉其他使用 GPU 的程式（遊戲、影片剪輯軟體等）
2. 減少批次大小，分批處理：
   ```cmd
   python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data --max_images 500
   ```
3. 如果還是不行，改用 CPU（會慢很多）：
   ```cmd
   python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data --device cpu
   ```

### 問題 2: "No module named 'torch'"

**原因**: PyTorch 未安裝或虛擬環境未啟動

**解決方法**：
1. 確認虛擬環境已啟動（命令提示符前有 `(venv_degradation)`）
2. 重新安裝 PyTorch（參考步驟 4）

### 問題 3: 處理速度很慢（遠慢於預期）

**檢查事項**：
1. 確認正在使用 GPU：
   ```cmd
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   必須顯示 `True`

2. 檢查是否有大量小尺寸影像（< 512x512）
   - 小影像的處理開銷較大

3. 查看 `nvidia-smi` 確認 GPU 使用率
   - 如果 GPU-Util 一直是 0%，表示沒有使用 GPU

### 問題 4: "ultralytics" 模型下載失敗

**錯誤**：網路連線問題導致 YOLOv8 模型下載失敗

**解決方法**：
1. 手動下載模型：
   - 網址: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
2. 將下載的 `yolov8n-seg.pt` 放到：
   ```
   C:\Users\你的使用者名稱\.cache\ultralytics\
   ```
3. 重新執行程式

### 問題 5: 中文路徑問題

**錯誤**：無法讀取包含中文的檔案路徑

**解決方法**：
1. 確保路徑使用反斜線 `\` 或雙反斜線 `\\`
2. 使用原始字串（raw string）：
   ```python
   r"C:\路徑\包含中文\影像.jpg"
   ```
3. 或避免在路徑中使用中文

## 效能優化建議 (Performance Tips)

### 1. 僅生成需要的降質類型

如果只需要某些降質類型，可以停用其他：

```cmd
# 只生成全局模糊
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data --no_object_blur --no_low_light

# 只生成低光照
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data --no_global_blur --no_object_blur
```

### 2. 分批處理

如果擔心處理中斷，可以分批處理：

```cmd
# 先處理 train
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_train --max_images 3069

# 再處理 val 和 test（需修改程式碼只處理特定分割）
```

### 3. 電源設定

確保 Windows 電源計畫設為「高效能」：
1. 控制台 → 硬體和音效 → 電源選項
2. 選擇「高效能」計畫
3. 或在「平衡」計畫中調整「處理器電源管理」為 100%

## 下一步 (Next Steps)

安裝和測試完成後：

1. **檢查降質效果**：
   - 隨機查看 10-20 對降質影像
   - 確認效果符合預期
   - 必要時調整參數

2. **開始訓練模型（Phase 3）**：
   - 使用生成的 `pairs.csv` 載入資料
   - 建構還原模型（如 U-Net, NAFNet）
   - 進行訓練和評估

3. **備份資料**：
   - 降質影像生成完成後記得備份
   - 避免重新處理浪費時間

## 技術支援 (Support)

如遇到問題：

1. 檢查錯誤訊息
2. 查看本文件的常見問題排解
3. 確認系統需求都符合
4. 使用 `--max_images 1` 測試單張影像

---

**最後更新**: 2025-11-04  
**適用系統**: Windows 10/11  
**GPU 需求**: NVIDIA RTX 4060 或其他支援 CUDA 的顯卡

