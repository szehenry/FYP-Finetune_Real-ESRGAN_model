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

## 📊 下一步：数据集分割 (Dataset Splitting)

在进行退化对生成之前，**必须先将数据集分割为 train/val/test**。

### 执行分割脚本

```bash
python3 split_dataset.py
```

### 为什么使用 Stratified Split？

**Stratified Split（分层分割）** 是一种确保各个子集具有相似数据分布的分割方法。

#### 三种分割方法对比

| 方法 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **Random Split** | 完全随机分配所有影像 | 简单 | ⚠️ 分布不均，大数据集可能占主导 |
| **Stratified Split** ✅ | 每个数据集按相同比例分割 | 确保多样性，分布均衡 | 需要知道数据来源 |
| **Group K-Fold** | 多次分割并平均结果 | 更可靠的评估 | 需要训练多次，耗时 |

#### Stratified Split 的工作原理

```
假设你有：
- Aerial-Traffic: 102 张原始（408 张含增强）
- FloodNet: 203 张原始（812 张含增强）
- VisDrone2019: 584 张原始（2336 张含增强）

❌ Random Split 的问题：
  可能结果：
  - Train: 80% 都是 VisDrone（因为它最多）
  - Test: 几乎没有 Aerial-Traffic（因为它最少）
  → 模型过度学习 VisDrone 的特征
  → 在其他数据集上表现差

✅ Stratified Split 的做法：
  每个数据集分别按 70/15/15 分割：
  
  Aerial-Traffic:
    Train: 71 张原始 (284 张含增强) → 70%
    Val: 16 张原始 (64 张含增强) → 15%
    Test: 15 张原始 (60 张含增强) → 15%
  
  FloodNet:
    Train: 142 张原始 (568 张含增强) → 70%
    Val: 30 张原始 (120 张含增强) → 15%
    Test: 31 张原始 (124 张含增强) → 15%
  
  VisDrone2019:
    Train: 409 张原始 (1636 张含增强) → 70%
    Val: 88 张原始 (352 张含增强) → 15%
    Test: 87 张原始 (348 张含增强) → 15%
  
  结果：每个数据集在 train/val/test 中都有代表
  → 模型学到所有场景的特征
  → 评估更可靠
```

#### 为什么 Stratified Split 更好？

1. **场景多样性保证**
   ```
   每个数据集代表不同的场景特征：
   - Aerial-Traffic: 交通场景，高速运动
   - FloodNet: 灾害场景，低光/噪点
   - VisDrone2019: 城市场景，密集物体
   
   Stratified Split 确保模型在训练时见到所有类型
   → 更好的泛化能力
   ```

2. **避免评估偏差**
   ```
   如果 Test set 只有某一种数据集：
   → 评估结果不能代表整体性能
   
   Stratified Split 确保 test set 包含所有场景
   → 更可靠的性能评估
   ```

3. **符合研究规范**
   ```
   学术界标准做法：
   - 多数据集融合训练时使用 Stratified Split
   - 可在论文中说明遵循最佳实践
   - 审稿人会认可这个方法
   ```

---

## 🎲 Random Seed 是什么？为什么需要它？

### Random Seed 基本概念

**Random Seed（随机种子）** 是用来初始化随机数生成器的一个数字。

```python
import random

# 不设置 seed：每次运行结果不同
random.shuffle([1, 2, 3, 4, 5])  # 可能是 [3, 1, 5, 2, 4]
random.shuffle([1, 2, 3, 4, 5])  # 可能是 [2, 4, 1, 5, 3]

# 设置 seed：每次运行结果相同
random.seed(42)
random.shuffle([1, 2, 3, 4, 5])  # 永远是 [5, 2, 4, 3, 1]
random.seed(42)
random.shuffle([1, 2, 3, 4, 5])  # 还是 [5, 2, 4, 3, 1]
```

### 为什么需要 Random Seed？

#### 1. **可重现性 (Reproducibility)**

```
没有 seed 的问题：
  第一次运行：train 包含 img_0.jpg
  第二次运行：train 包含 img_100.jpg
  
  结果：
  - 无法重现实验结果
  - 无法比较不同方法（因为数据分割不同）
  - 论文中的结果别人无法验证

有 seed 的好处：
  每次运行：train 永远包含相同的影像
  
  结果：
  ✅ 实验可重现
  ✅ 可以公平比较不同模型
  ✅ 论文结果可验证
```

#### 2. **调试和开发**

```
场景：你发现模型在某些影像上表现不好

没有 seed：
  - 重新分割后那些影像可能在不同集合
  - 无法追踪问题
  - 浪费时间

有 seed：
  - 分割永远相同
  - 可以精确定位问题影像
  - 快速迭代改进
```

#### 3. **学术规范**

```
论文中必须说明：
  "We split the dataset using stratified random sampling 
   with a fixed random seed (seed=42) to ensure reproducibility."

没有 seed → 审稿人会质疑结果的可靠性
有 seed → 符合研究标准
```

### Random Seed 的选择

**常见的选择：**
- `42` - 最流行（来自《银河系漫游指南》）
- `0` - 简单直接
- `2024` - 年份
- 任何固定数字都可以

**重要：**
- ✅ 选择一个数字并**保持不变**
- ❌ 不要随意更改 seed（除非有特殊原因）
- ✅ 在论文/报告中明确说明使用的 seed

### 在本专案中的应用

```python
# split_dataset.py 中的配置
RANDOM_SEED = 42

# 这确保：
✅ 每次运行分割脚本得到相同的 train/val/test
✅ 你和导师可以重现相同的结果
✅ 如果需要重新训练，数据分割保持一致
✅ 论文中可以说明使用了固定 seed
```

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

