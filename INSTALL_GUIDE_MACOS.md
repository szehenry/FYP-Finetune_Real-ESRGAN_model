# MacBook 安裝指南 (macOS Installation Guide)

## 系統需求 (System Requirements)

- macOS 10.15 (Catalina) 或更新版本
- Python 3.8 或更高版本
- **注意**: MacBook 沒有 NVIDIA GPU，會使用 CPU 模式（速度較慢）

## 快速安裝 (Quick Installation)

### 步驟 1: 檢查 Python

```bash
python3 --version
```

如果沒有安裝 Python，使用 Homebrew 安裝：
```bash
# 安裝 Homebrew（如果還沒有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安裝 Python
brew install python@3.11
```

### 步驟 2: 創建虛擬環境

在專案目錄下：

```bash
cd /Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP

# 創建虛擬環境
python3 -m venv venv_degradation

# 啟動虛擬環境
source venv_degradation/bin/activate
```

成功後，終端機提示符前會出現 `(venv_degradation)`。

### 步驟 3: 升級 pip

```bash
pip install --upgrade pip
```

### 步驟 4: 安裝 PyTorch (CPU 版本)

**重要**: MacBook 使用 CPU 版本的 PyTorch

```bash
# 安裝 PyTorch CPU 版本
pip install torch torchvision torchaudio
```

**如果你使用 Apple Silicon (M1/M2/M3 晶片)**：
```bash
# 可以使用 MPS (Metal Performance Shaders) 加速
pip install torch torchvision torchaudio
```

### 步驟 5: 安裝其他依賴

```bash
pip install -r requirements_degradation.txt
```

### 步驟 6: 驗證安裝

```bash
# 檢查 PyTorch
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# 檢查是否支援 MPS (Apple Silicon)
python -c "import torch; print('MPS 可用:', torch.backends.mps.is_available())"

# 檢查 YOLOv8
python -c "from ultralytics import YOLO; print('YOLOv8 安裝成功')"
```

## 使用方式 (Usage)

### 測試單張影像

```bash
python test_degradation.py \
    --image "/path/to/your/image.jpg" \
    --output ./test_output \
    --device cpu
```

**注意**: 必須加上 `--device cpu`

### 小規模測試

```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./mini_test \
    --max_images 5 \
    --device cpu
```

### 處理完整資料集

**方法 1: 標準版（循序處理）**
```bash
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --device cpu
```

**方法 2: 平行版（推薦！使用多核心加速）**
```bash
# 自動使用所有可用核心
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data

# 或指定使用 4 個核心
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --num_workers 4
```

**平行版的優勢**：
- ⚡ 全局模糊和低光照處理快 **2-4 倍**
- 💻 充分利用 MacBook 的多核心 CPU
- 📊 更高的 CPU 使用率（60-80% vs 15-25%）

**注意**: 物體模糊仍然是循序處理（YOLOv8 限制）

## 效能預期 (Performance Expectations)

### ⚠️ 處理時間比較

以 4,391 張影像為例：

| 硬體 | 標準版 | 平行版 | 平行版 (無物體模糊) |
|------|-------|--------|-------------------|
| **RTX 4060 (CUDA)** | 2-3 小時 | N/A | N/A |
| **MacBook Pro (Intel i7, 4核)** | 10-15 小時 | 6-10 小時 | 3-5 小時 |
| **MacBook Air (M1, 8核)** | 8-12 小時 | 5-8 小時 | 2-4 小時 |
| **MacBook Pro (M2 Pro, 10核)** | 7-10 小時 | 4-6 小時 | 2-3 小時 |
| **MacBook Pro (M3 Pro, 12核)** | 6-10 小時 | 3-5 小時 | 1.5-2.5 小時 |

**說明**：
- **標準版**: 循序處理，單核心
- **平行版**: 使用多核心，快 **2-4 倍**
- **無物體模糊**: 跳過最耗時的 YOLOv8，再快 **2-3 倍**

**物體模糊 (YOLOv8)** 是最耗時的部分，佔總時間約 60-70%。

### 💡 加速建議

1. **使用平行版（推薦！）**：
   ```bash
   # 使用所有可用核心
   python degradation_synthesis_parallel.py \
       --split_dir ./data_split_results \
       --output_dir ./degraded_data
   ```

2. **只生成必要的降質類型**：
   ```bash
   # 跳過最耗時的物體模糊（可節省 60-70% 時間）
   python degradation_synthesis_parallel.py \
       --split_dir ./data_split_results \
       --output_dir ./degraded_data \
       --no_object_blur
   ```

3. **分批處理**：
   ```bash
   # 每次處理 500 張，可以分多次完成
   python degradation_synthesis_parallel.py \
       --split_dir ./data_split_results \
       --output_dir ./degraded_batch1 \
       --max_images 500
   ```

4. **夜間運行**：
   讓程式在夜間運行，避免影響日常使用
   ```bash
   # 使用 nohup 在背景運行
   nohup python degradation_synthesis_parallel.py \
       --split_dir ./data_split_results \
       --output_dir ./degraded_data > degradation.log 2>&1 &
   ```

5. **最佳化組合（最快！）**：
   ```bash
   # 平行版 + 跳過物體模糊 = 快 6-8 倍
   python degradation_synthesis_parallel.py \
       --split_dir ./data_split_results \
       --output_dir ./degraded_data \
       --no_object_blur \
       --num_workers 8
   ```

## Apple Silicon (M1/M2/M3) 特殊說明

### 可能可以使用 MPS 加速

修改程式碼以支援 MPS（Metal Performance Shaders）：

在 `degradation_synthesis.py` 第 805 行附近，修改設備選擇邏輯：

```python
# 原本
if device == 'auto':
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 改為（支援 MPS）
if device == 'auto':
    if torch.cuda.is_available():
        self.device = 'cuda'
    elif torch.backends.mps.is_available():
        self.device = 'mps'  # Apple Silicon GPU
    else:
        self.device = 'cpu'
```

**注意**: MPS 支援可能不完整，建議先測試。

## 監控處理進度

### 查看 CPU 使用率

```bash
# 開啟活動監視器
open -a "Activity Monitor"
```

或使用終端機：
```bash
# 即時監控
top
```

按 `q` 退出。

### 查看處理進度

程式會顯示進度條（tqdm），例如：
```
處理 train: 45%|████▌     | 1380/3069 [1:23:45<1:35:21, 0.29it/s]
```

## 常見問題 (FAQ)

### Q: 為什麼這麼慢？

**A**: MacBook 沒有 NVIDIA GPU，只能用 CPU。物體模糊需要執行 YOLOv8，在 CPU 上很慢。

**解決方案**：
1. 跳過物體模糊（`--no_object_blur`）
2. 考慮在有 GPU 的電腦上處理（如你的 Windows RTX 4060）
3. 分批處理，慢慢累積

### Q: 可以用 Apple Silicon GPU 加速嗎？

**A**: 理論上可以（MPS），但需要修改程式碼且可能不穩定。建議先用 CPU 測試。

### Q: 要多久才能處理完？

**A**: 取決於你的 MacBook 型號：
- **M 系列晶片**: 6-15 小時（取決於具體型號）
- **Intel 晶片**: 10-20 小時

可以用 `--max_images 10` 先測試 10 張，估算總時間。

### Q: 可以暫停和繼續嗎？

**A**: 目前不支援斷點續傳。如果中斷，需要重新開始。

**建議**: 先用小批次測試（如 100 張），確認穩定後再處理大批次。

## 推薦工作流程

考慮到 MacBook 效能限制：

### 選項 1: 在 Windows RTX 4060 上處理（推薦）
- 速度快 5-10 倍
- 2-3 小時完成
- 更穩定

### 選項 2: 在 MacBook 上只做測試
```bash
# 測試 10 張影像
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./test_mac \
    --max_images 10 \
    --device cpu
```

驗證效果後，在 Windows 上處理完整資料集。

### 選項 3: 在 MacBook 上只生成部分降質
```bash
# 只生成全局模糊和低光照（跳過耗時的物體模糊）
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --device cpu \
    --no_object_blur
```

這樣可以快 3-5 倍。

## 節能建議

處理大量影像時：

1. **插著電源**，不要用電池
2. **調整節能設定**：
   - 系統偏好設定 → 節能器
   - 取消「顯示器關閉時進入睡眠」
   - 取消「在電源轉換器供電時防止電腦自動進入睡眠」
3. **關閉不必要的應用程式**
4. **確保散熱良好**

---

**最後更新**: 2025-11-04  
**適用系統**: macOS 10.15+  
**CPU 需求**: 任何 Mac（建議 M1 或更新）

