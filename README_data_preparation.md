# 資料準備腳本使用指南 (Data Preparation Quick Start)

## ⚠️ 重要須知（請先閱讀）

### 安全保證
- ✅ **絕對不會刪除任何影像檔案**
- ✅ 步驟 1-2 只分析並生成建議報告
- ✅ 步驟 3 只會新增增強影像，不會修改原始檔案
- ✅ 所有原始影像都會完整保留

### 標籤/Metadata 自動處理
- ✅ **不需要重新跑 image_curator**
- ✅ 增強影像會自動繼承原始影像的所有標籤
- ✅ 保存在 `augmented_metadata_<dataset>.csv`
- ✅ 可直接用於後續訓練

### 當前配置（擴充倍率：4×）
```
每張原始影像產生：
- 水平翻轉：1 張
- 隨機裁切：2 張
總共：3 張增強影像（加上原始 = 4×）
```

---

## 🎯 快速開始

### 1. 確認環境

```bash
# 使用你當前的環境
source image_curator_env/bin/activate

# 或使用另一個環境
# source venv_auto_curator/bin/activate

# 確認套件（應該都已安裝）
pip list | grep -E "(opencv|numpy|pandas)"
```

### 2. 執行腳本

```bash
cd /Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP
python3 data_preparation.py
```

### 3. 查看結果

執行完成後，結果會保存在 `data_preparation_results/` 資料夾中。

---

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `data_preparation.py` | 主要執行腳本（包含三個步驟） |
| `資料準備流程說明.md` | **完整技術文件**（用於報告撰寫） |
| `README_data_preparation.md` | 本檔案（快速使用指南） |

---

## 🔍 執行步驟詳解

腳本會自動執行三個步驟：

### 步驟 1️⃣：影像品質與多樣性分析
- 分析每張影像的清晰度、邊緣密度、亮度等指標
- 產生資料集多樣性報告
- **輸出**：`quality_analysis_*.csv`, `diversity_report.json`

### 步驟 2️⃣：偵測重複/相似影像
- 使用 Perceptual Hashing 找出相似影像
- 提供移除建議（**建議手動檢查後再刪除**）
- **輸出**：`similar_pairs_*.csv`, `removal_suggestions_*.csv`

### 步驟 3️⃣：資料增強
- 對每張影像進行翻轉、旋轉、裁切
- 產生約 **8-9 倍**的擴充資料
- **輸出**：增強影像保存在原資料夾，日誌記錄在 `augmentation_log_*.csv`

---

## 📊 預期結果（已調整為 3,000 張目標）

```
原始影像：~900 張
↓
移除重複：~800 張（建議手動確認）
↓
資料增強：~3,200 張（800 × 4）
↓
退化對生成：~6,400 對（3,200 × 2: motion blur + low-light）
```

**✅ 控制訓練規模，避免訓練時間過長**

---

## ⚙️ 調整參數

如需調整增強倍率或相似度閾值，請編輯 `data_preparation.py` 中的配置區：

```python
# 資料增強參數（當前配置：4× 擴充）
AUGMENTATION_CONFIG = {
    "enable_flip": True,           # 水平翻轉 (+1)
    "enable_rotation": False,      # 旋轉（已關閉以控制數量）
    "enable_crop": True,           # 裁切
    "crop_size": 512,              # 裁切大小
    "num_crops_per_image": 2,      # 每張影像裁切數量 (+2)
}

# 如果想要更多資料，可以：
# - 增加 num_crops_per_image 到 3-5
# - 啟用 enable_rotation: True（會額外產生 3 張）

# 相似度檢測參數
SIMILARITY_CONFIG = {
    "phash_threshold": 5,          # 閾值越小越嚴格
}
```

---

## ⚠️ 重要提醒

1. **❌ 絕對不會刪除任何影像**：
   - 腳本只會「建議」移除重複影像
   - 所有原始影像都會保留
   - 請手動檢查 `removal_suggestions_*.csv` 後再決定是否刪除
   - **步驟 1-2 只分析，步驟 3 只新增檔案**

2. **🏷️ 增強影像的標籤/Metadata**：
   - ✅ **自動生成**，不需要重新跑 image_curator
   - 自動繼承原始影像的所有標籤（scene_type, has_text, is_night 等）
   - 保存在 `augmented_metadata_*.csv`
   - 可直接用於後續訓練

3. **增強影像保存位置**：
   - 與原始影像在同一資料夾
   - 檔名格式：`原檔名_方法.jpg`（如 `img_0_flip_h.jpg`）

4. **執行時間**：
   - 約 25-45 分鐘（視影像數量）
   - 可以隨時中斷，已處理的結果會保存

5. **使用環境**：
   - ✅ 可以使用 `image_curator_env`
   - ✅ 或使用 `venv_auto_curator`
   - 兩者都有需要的套件（opencv, numpy, pandas）

---

## 📖 詳細技術文件

完整的技術說明、演算法原理、理論基礎請參考：
**`資料準備流程說明.md`**

該文件包含：
- ✅ 每個步驟的詳細原理
- ✅ 數學公式和演算法說明
- ✅ 為什麼資料增強有效的深入解釋
- ✅ 所有參數的建議值
- ✅ 適合用於 FYP 報告撰寫

---

## 🐛 問題排除

### 問題：找不到影像檔案
**解決**：確認 `FYP_Images` 資料夾路徑正確

### 問題：記憶體不足
**解決**：腳本採用逐張處理，記憶體需求很低。如仍有問題，可分批處理資料集

### 問題：執行時間太長
**解決**：
- 步驟 2（相似度檢測）最耗時
- 可以暫時停用：修改程式碼跳過步驟 2

---

## 📧 需要協助？

如有任何問題，請檢查：
1. `資料準備流程說明.md` 的「常見問題」章節
2. 執行過程中的錯誤訊息
3. `data_preparation_results/` 中的輸出檔案

---

**Good Luck with Your FYP! 🚀**

