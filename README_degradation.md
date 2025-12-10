# 影像降質合成工具 (Image Degradation Synthesis Tool)

## 概述 (Overview)

這是 Phase 2 的影像降質合成工具，用於從乾淨影像生成訓練用的降質影像。支援三種降質類型：

1. **全局運動模糊** (Global Motion Blur) - 模擬相機運動造成的整體模糊
2. **物體運動模糊** (Object Motion Blur) - 僅對特定物體（如車輛）應用運動模糊
3. **低光照降質** (Low-light Degradation) - 模擬低光環境下的影像品質下降

## 功能特色 (Features)

✅ **GPU 加速** - 支援 NVIDIA GPU (CUDA)，在 RTX 4060 上運行速度快  
✅ **智慧物體偵測** - 使用 YOLOv8-seg 自動識別車輛等物體  
✅ **真實降質模擬** - 基於物理的降質模型  
✅ **完整元數據** - 記錄每個降質影像的參數  
✅ **批次處理** - 支援大規模資料集處理  
✅ **進度追蹤** - 即時顯示處理進度  

## 系統需求 (System Requirements)

### 硬體 (Hardware)
- **建議**: NVIDIA GPU (如 RTX 4060) 搭配 CUDA
- **最低**: CPU（速度較慢）
- **RAM**: 至少 8GB（16GB 以上更佳）
- **硬碟空間**: 根據資料集大小，建議預留原始資料 3-5 倍的空間

### 軟體 (Software)
- Windows 10/11
- Python 3.8 或更高版本
- CUDA 11.8 或更高（如使用 GPU）

## 安裝步驟 (Installation)

### 1. 創建虛擬環境 (Create Virtual Environment)

```bash
# 使用 Python 內建 venv
python -m venv venv_degradation

# 啟動虛擬環境 (Windows)
venv_degradation\Scripts\activate

# 啟動虛擬環境 (Mac/Linux)
source venv_degradation/bin/activate
```

### 2. 安裝依賴套件 (Install Dependencies)

```bash
# 安裝 PyTorch (GPU 版本 - 適用於 RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements_degradation.txt
```

**注意**: 如果你的電腦沒有 NVIDIA GPU，請改用 CPU 版本：
```bash
pip install torch torchvision torchaudio
```

### 3. 驗證安裝 (Verify Installation)

```python
# 測試 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 測試 YOLOv8
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

## 使用方法 (Usage)

### 🚀 兩種處理模式

我們提供兩個版本：

| 版本 | 檔案 | 適用情況 | 速度 |
|------|------|---------|------|
| **標準版** | `degradation_synthesis.py` | GPU 處理 (CUDA) | ⭐⭐⭐⭐⭐ 最快 |
| **平行版** | `degradation_synthesis_parallel.py` | 多核心 CPU | ⭐⭐⭐ 快 2-4 倍 |

**選擇建議**：
- ✅ **有 NVIDIA GPU (如 RTX 4060)**: 使用標準版 + GPU
- ✅ **只有 CPU (如 MacBook)**: 使用平行版 + 多核心
- ✅ **測試用途**: 使用標準版，簡單直接

### 基本使用 - 標準版 (Basic Usage - Standard)

處理完整資料集，生成所有三種降質類型：

```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data
```

### 基本使用 - 平行版 (Basic Usage - Parallel)

使用所有可用 CPU 核心平行處理（適合 MacBook）：

```bash
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data
```

指定使用 4 個工作處理器：

```bash
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --num_workers 4
```

### 測試運行 (Test Run)

先用少量影像測試（建議第一次使用）：

```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./test_degraded \
    --max_images 10
```

這會從每個分割（train/val/test）中各處理 10 張影像。

### 進階選項 (Advanced Options)

#### 1. 選擇特定降質類型 (Select Specific Degradation Types)

只生成全局模糊和低光照，跳過物體模糊：
```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --no_object_blur
```

只生成物體模糊：
```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --no_global_blur \
    --no_low_light
```

#### 2. 指定計算設備 (Specify Device)

強制使用 CPU（如果 GPU 有問題）：
```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --device cpu
```

#### 3. 處理部分資料 (Process Subset)

處理每個分割的前 100 張影像：
```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --max_images 100
```

### 完整參數說明 (Complete Arguments)

```
必要參數 (Required):
  --split_dir PATH        資料分割目錄，包含 train_list.txt, val_list.txt, test_list.txt
  --output_dir PATH       輸出目錄路徑

可選參數 (Optional):
  --device {auto,cuda,cpu}  計算設備 (預設: auto)
  --max_images N           每個分割最大處理影像數，用於測試 (預設: None，處理全部)
  --no_global_blur         停用全局運動模糊
  --no_object_blur         停用物體運動模糊
  --no_low_light           停用低光照降質
```

## 輸出結構 (Output Structure)

處理完成後，輸出目錄結構如下：

```
degraded_data/
├── degraded/                    # 降質影像目錄
│   ├── img_001_global_blur.jpg
│   ├── img_001_object_blur.jpg
│   ├── img_001_low_light.jpg
│   ├── img_002_global_blur.jpg
│   └── ...
├── metadata/                    # 元數據目錄
│   ├── img_001_global_blur.json
│   ├── img_001_object_blur.json
│   ├── img_001_low_light.json
│   └── ...
├── pairs.csv                    # 配對資訊 CSV
└── degradation_summary.json     # 處理摘要
```

### pairs.csv 格式

| degraded_path | target_path | split | mode | metadata_path |
|--------------|-------------|-------|------|---------------|
| degraded/img_001_global_blur.jpg | /path/to/original/img_001.jpg | train | global_blur | metadata/img_001_global_blur.json |
| degraded/img_001_low_light.jpg | /path/to/original/img_001.jpg | train | low_light | metadata/img_001_low_light.json |

這個 CSV 檔案可以直接用於訓練模型的資料載入器 (dataloader)。

### 元數據 JSON 範例

**全局模糊 (Global Blur)**:
```json
{
  "mode": "global_motion_blur",
  "kernel_size": 15,
  "angle": 45.3,
  "motion_length": 12.5,
  "use_rolling_shutter": false,
  "shear_amount": 0.0
}
```

**物體模糊 (Object Blur)**:
```json
{
  "mode": "object_motion_blur",
  "objects_detected": 2,
  "angle": 135.7,
  "motion_length": 10,
  "feather_amount": 5,
  "conf_threshold": 0.25,
  "detected_classes": ["car", "truck"]
}
```

**低光照 (Low-light)**:
```json
{
  "mode": "low_light",
  "exposure_factor": 0.35,
  "wb_shift_b": 1.05,
  "wb_shift_g": 1.02,
  "wb_shift_r": 0.88,
  "black_level": 0.05,
  "highlight_rolloff": 0.9,
  "shot_noise_scale": 0.01,
  "read_noise_std": 5.2,
  "jpeg_quality": 75,
  "use_jpeg": true
}
```

## 降質類型詳解 (Degradation Types Explained)

### 1. 全局運動模糊 (Global Motion Blur)

**原理**: 模擬相機整體運動（如手持拍攝抖動）造成的模糊效果。

**技術細節**:
- 生成軌跡式的點擴散函數 (PSF - Point Spread Function)
- 使用 FFT 快速卷積應用模糊
- 可選的捲簾快門效果 (rolling shutter)

**參數**:
- `kernel_size`: 模糊核大小（固定 15）
- `angle`: 運動方向 0-360° (隨機)
- `motion_length`: 運動距離 5-25 像素 (隨機)
- `use_rolling_shutter`: 是否加入捲簾效果 (預設關閉)

### 2. 物體運動模糊 (Object Motion Blur)

**原理**: 僅對移動的物體（如行駛中的車輛）應用運動模糊，背景保持清晰。

**技術細節**:
- 使用 YOLOv8-seg 偵測並分割物體
- 支援的物體類別：car, truck, bus, motorcycle, bicycle, person
- 方向性模糊核應用於物體區域
- 邊緣羽化避免光暈 (halo artifacts)
- 智慧合成回背景

**參數**:
- `conf_threshold`: YOLO 信心度閾值 (預設 0.25)
- `angle`: 運動方向 (隨機)
- `motion_length`: 運動距離 5-15 像素 (隨機)
- `feather_amount`: 邊緣羽化量 5 像素

**注意**: 如果影像中沒有偵測到物體，將不會生成此類型的降質影像。

### 3. 低光照降質 (Low-light Degradation)

**原理**: 模擬低光環境下的影像品質下降，包括曝光不足、噪聲增加、色彩偏移等。

**技術細節**:
1. **降低曝光**: 乘以因子 0.2-0.5
2. **白平衡偏移**: 模擬色溫變化（偏冷色調）
3. **色調曲線調整**: 
   - 黑色提升 (black level lift) - 陰影變灰
   - 高光壓縮 (highlight roll-off) - 亮部細節損失
4. **噪聲添加**:
   - 散粒噪聲 (shot noise) - 與亮度相關
   - 讀取噪聲 (read noise) - 常數噪聲
5. **JPEG 壓縮**: 品質 60-85 (模擬相機內部壓縮)

**參數**:
- `exposure_factor`: 0.2-0.5 (隨機)
- `wb_shift`: RGB 通道分別調整
- `shot_noise_scale`: 0.01
- `read_noise_std`: 3-8 (隨機)
- `jpeg_quality`: 60-85 (隨機)

## 效能建議 (Performance Tips)

### 使用 GPU 加速

確保 PyTorch 能正確使用你的 RTX 4060：

```python
# 在程式執行前檢查
import torch
print(torch.cuda.is_available())  # 應該是 True
print(torch.cuda.get_device_name(0))  # 應該顯示 "NVIDIA GeForce RTX 4060"
```

### 預期處理速度 (Expected Speed)

在 RTX 4060 上的大致速度（依影像大小而異）：

- **全局模糊**: ~0.1-0.3 秒/張
- **物體模糊**: ~0.3-0.8 秒/張 (因為需要 YOLO 推論)
- **低光照**: ~0.1-0.2 秒/張

例如，處理 4000 張影像（生成 3 種降質 = 12000 張）：
- **GPU**: 約 1-2 小時
- **CPU**: 約 6-12 小時

### 記憶體管理

如果遇到 GPU 記憶體不足 (Out of Memory)：

1. 檢查是否有其他程式佔用 GPU
2. 嘗試處理較小批次（使用 `--max_images`）
3. 考慮使用 CPU（`--device cpu`）

## 常見問題 (FAQ)

### Q1: YOLOv8 第一次運行很慢？

**A**: 第一次運行時，YOLOv8 會自動下載預訓練模型（約 6MB），這是正常的。下載完成後會快很多。

### Q2: 物體模糊沒有生成影像？

**A**: 這通常表示影像中沒有偵測到車輛等物體。你可以：
- 降低 `conf_threshold`（但可能增加誤偵測）
- 檢查影像內容是否確實包含可偵測物體
- 查看 metadata 中的 `objects_detected` 欄位

### Q3: 處理速度很慢？

**A**: 檢查：
- 是否正確使用 GPU？運行 `nvidia-smi` 查看 GPU 使用情況
- 是否有大量小尺寸影像？處理開銷可能較大
- 嘗試只啟用特定降質類型進行測試

### Q4: 可以自訂降質參數嗎？

**A**: 目前參數是隨機生成的。如果需要固定參數，可以修改 `synthesize()` 方法中的預設值。未來版本可能會加入配置檔案支援。

### Q5: 如何檢查生成的降質效果？

**A**: 你可以：
1. 隨機挑選幾張降質影像用影像檢視器打開
2. 對照原始影像查看降質效果
3. 閱讀對應的 JSON 元數據了解參數

範例程式碼：
```python
import cv2
import json

# 讀取降質影像和原始影像
degraded = cv2.imread('degraded_data/degraded/img_001_global_blur.jpg')
original = cv2.imread('/path/to/original/img_001.jpg')

# 讀取元數據
with open('degraded_data/metadata/img_001_global_blur.json', 'r') as f:
    params = json.load(f)
    print(params)

# 並排顯示
import numpy as np
comparison = np.hstack([original, degraded])
cv2.imshow('Original vs Degraded', comparison)
cv2.waitKey(0)
```

## 下一步 (Next Steps)

降質影像生成完成後，你可以：

1. **檢查輸出**: 確認降質效果符合預期
2. **訓練模型**: 使用 `pairs.csv` 載入資料訓練還原模型
3. **調整參數**: 根據初步結果調整降質強度
4. **擴充降質類型**: 添加其他類型如雨滴、霧霾等（Phase 3）

## 技術支援 (Support)

如果遇到問題：

1. 檢查 Python 版本和套件版本
2. 確認 CUDA 和 PyTorch 相容性
3. 查看錯誤訊息和日誌
4. 使用 `--max_images 1` 測試單張影像

## 授權 (License)

此工具為學術研究項目的一部分。

---

**作者**: AI Assistant  
**日期**: 2025-11-04  
**版本**: 1.0.0

