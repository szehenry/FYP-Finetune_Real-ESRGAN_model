# 降質合成快速入門 (Quick Start Guide)

## 5 分鐘快速開始 (5-Minute Quick Start)

### 前置要求 (Prerequisites)
- ✅ Windows 10/11
- ✅ Python 3.8+
- ✅ NVIDIA RTX 4060 + 最新驅動程式

### 步驟 1: 安裝 (2 分鐘)

開啟 PowerShell/CMD 在專案資料夾：

```cmd
# 創建並啟動虛擬環境
python -m venv venv_degradation
venv_degradation\Scripts\activate

# 安裝 PyTorch (GPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他套件
pip install -r requirements_degradation.txt
```

### 步驟 2: 測試單張影像 (1 分鐘)

```cmd
# 用一張影像快速測試
python test_degradation.py --image "你的影像路徑.jpg" --output .\quick_test
```

查看 `quick_test\comparison_grid.jpg` 確認效果。

### 步驟 3: 小規模測試 (1 分鐘)

```cmd
# 處理 5 張影像測試完整流程
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\mini_test --max_images 5
```

### 步驟 4: 正式處理 (2-3 小時)

**Windows (RTX 4060) - 使用 GPU**：
```cmd
# 處理完整資料集（GPU 加速）
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data
```

**Mac 或只有 CPU - 使用平行處理**：
```bash
# 使用多核心 CPU 加速（快 2-4 倍）
python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./degraded_data
```

完成！🎉

---

## 詳細指南 (Detailed Guides)

需要更多資訊？查看：

- 📦 **完整安裝**: `INSTALL_GUIDE_DEGRADATION_WINDOWS.md`
- 📖 **使用說明**: `README_degradation.md`
- 🔧 **排錯**: `INSTALL_GUIDE_DEGRADATION_WINDOWS.md` → 常見問題排解

---

## 常用指令參考 (Common Commands)

### 選擇處理模式

**GPU 模式（Windows RTX 4060）**：
```cmd
python degradation_synthesis.py --split_dir .\data_split_results --output_dir .\degraded_data
```

**CPU 平行模式（Mac 或多核心 CPU）**：
```bash
# 自動使用所有核心
python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./degraded_data

# 指定使用 4 個核心
python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./degraded_data --num_workers 4
```

### 只生成特定降質類型

```bash
# 只要全局模糊（標準版）
python degradation_synthesis.py --split_dir ./data_split_results --output_dir ./only_blur --no_object_blur --no_low_light

# 只要低光照（平行版，快 2-4 倍）
python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./only_lowlight --no_global_blur --no_object_blur
```

### 分批處理

```bash
# 處理 1000 張（平行版）
python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./batch1 --max_images 1000
```

### 最快組合（跳過物體模糊 + 平行處理）

```bash
# Mac 用戶推薦：快 6-8 倍！
python degradation_synthesis_parallel.py --split_dir ./data_split_results --output_dir ./degraded_data --no_object_blur
```

---

## 檢查清單 (Checklist)

在開始前確認：

- [ ] Python 已安裝且版本 ≥ 3.8
- [ ] NVIDIA 驅動程式已安裝
- [ ] `nvidia-smi` 能正確顯示 GPU 資訊
- [ ] PyTorch CUDA 可用（`torch.cuda.is_available()` 為 `True`）
- [ ] 有足夠硬碟空間（至少 10 GB）
- [ ] 已用少量影像測試過

---

## 預期結果 (Expected Results)

### 處理統計 (Processing Statistics)

根據你的資料集（4391 張影像）：

| 項目 | 數量 |
|------|------|
| 原始影像 | 4,391 張 |
| 降質影像（估計）| ~10,000-13,000 張 |
| 處理時間（RTX 4060）| 2-3 小時 |
| 硬碟空間 | 5-10 GB |

### 輸出檔案 (Output Files)

```
degraded_data/
├── degraded/                 # 約 10,000+ 張降質影像
│   ├── img_001_global_blur.jpg
│   ├── img_001_object_blur.jpg
│   ├── img_001_low_light.jpg
│   └── ...
├── metadata/                 # 對應的 JSON 元數據
│   ├── img_001_global_blur.json
│   ├── img_001_object_blur.json
│   ├── img_001_low_light.json
│   └── ...
├── pairs.csv                # ⭐ 訓練用配對清單
└── degradation_summary.json # 處理摘要
```

**最重要**: `pairs.csv` 包含所有降質-乾淨影像配對，可直接用於訓練。

---

## 下一步 (Next Steps)

✅ 降質合成完成後：

1. **驗證資料品質**
   - 隨機檢查 20-30 對影像
   - 確認降質效果真實

2. **準備訓練**
   - 使用 `pairs.csv` 建構 PyTorch Dataset
   - 設計還原模型（如 U-Net）
   - 開始訓練！

3. **實驗不同參數**
   - 如果降質太強/太弱，可調整參數
   - 編輯 `degradation_synthesis.py` 中的預設值

---

## 獲取幫助 (Get Help)

遇到問題？

1. ❓ 查看 `README_degradation.md` 的 FAQ 章節
2. 🔧 查看 `INSTALL_GUIDE_DEGRADATION_WINDOWS.md` 排錯章節
3. 🧪 用 `--max_images 1` 測試單張影像找出問題

---

**Good luck! 祝你專案順利！** 🚀

