# 圖像篩選工具安裝和使用指南

## 🚀 快速開始

### 1. 環境設置（推薦使用虛擬環境）

```bash
# 進入項目目錄
cd "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP"

# 創建虛擬環境
python -m venv image_curator_env

# 激活虛擬環境
source image_curator_env/bin/activate
```

### 2. 安裝依賴

#### 基礎依賴（必需）
```bash
pip install opencv-python pandas numpy pathlib2
```

#### OCR功能（可選但推薦）
```bash
# 安裝PaddlePaddle後端
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安裝PaddleOCR
pip install paddleocr
```

### 3. 運行工具

#### 不使用OCR（基礎版本）
```bash
python image_curator.py
```

#### 使用OCR（推薦）
```bash
python image_curator.py --ocr
```

## 📋 工作流程

### 推薦操作順序
1. **查看圖像** - 程序自動顯示
2. **設置標籤** - 按需要設置：
   - 按 `t` 切換文本標籤
   - 按 `1-7` 選擇場景類型
3. **接受圖像** - 按 `a` 接受並自動保存
4. **或拒絕圖像** - 按 `r` 拒絕並跳過

### 鍵盤控制
| 按鍵 | 功能 | 說明 |
|------|------|------|
| `a` | 接受圖像 | 保存當前標籤並跳到下一張 |
| `r` | 拒絕圖像 | 跳過當前圖像 |
| `t` | 切換文本標籤 | 是/否 |
| `1-7` | 場景類型 | 可多選 |
| `n/b` | 前進/後退 | 瀏覽圖像 |
| `s` | 手動保存 | 需先按 'a' 接受 |
| `q` | 退出 | 保存進度並退出 |
| `h` | 幫助 | 顯示控制說明 |

### 場景類型對照
| 數字 | 場景類型 | 說明 |
|------|----------|------|
| `1` | road | 道路場景 |
| `2` | building | 建築物 |
| `3` | infrastructure | 基礎設施 |
| `4` | vehicle | 車輛 |
| `5` | vegetation | 植被 |
| `6` | low-light | 低光照 |
| `7` | other | 其他 |

## 🔧 故障排除

### 常見問題及解決方案

#### 1. PaddleOCR相關錯誤
```bash
# 問題：ModuleNotFoundError: No module named 'paddle'
# 解決：安裝PaddlePaddle
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

# 問題：use_angle_cls參數棄用警告
# 解決：已在代碼中修復，使用use_textline_orientation參數
```

#### 2. OpenCV顯示問題
```bash
# 問題：圖像窗口無法顯示
# 解決：確保在有GUI的環境中運行，或使用遠程桌面
```

#### 3. 權限問題
```bash
# 問題：無法寫入CSV文件
# 解決：檢查輸出目錄的寫入權限
chmod 755 /path/to/output/directory
```

#### 4. 圖像文件夾不存在
```bash
# 問題：錯誤: 圖像文件夾不存在
# 解決：檢查路徑是否正確
python image_curator.py --root /correct/path/to/images
```

### 性能優化建議

1. **大量圖像處理**
   - 分批處理，每次處理1000-2000張
   - 定期備份CSV文件

2. **OCR性能**
   - OCR會增加處理時間，但提高標註準確性
   - 可以先不使用OCR快速篩選，再用OCR精確標註

3. **內存使用**
   - 程序會自動調整圖像大小以適應屏幕
   - 如遇內存問題，可重啟程序（支持斷點續傳）

## 📊 輸出格式

### CSV文件結構
```csv
filepath,has_text,scene_type,timestamp
/path/to/image1.jpg,1,"road,vehicle",2024-10-24T10:30:00
/path/to/image2.jpg,0,building,2024-10-24T10:31:00
```

### 字段說明
- `filepath`: 圖像文件完整路徑
- `has_text`: 是否包含文本（0=否，1=是）
- `scene_type`: 場景類型（逗號分隔的多個類型）
- `timestamp`: 標註時間戳

## 🔄 斷點續傳功能

- 程序會自動檢測已處理的圖像
- 重新運行時從未處理的圖像開始
- 可以隨時按 `q` 退出，下次運行時繼續
- CSV文件實時保存，避免數據丟失

## 💡 最佳實踐

### 篩選標準
- **接受的圖像**：清晰、低噪聲、視角清楚
- **拒絕的圖像**：模糊、噪聲重、壓縮伪影明顯

### 標註建議
1. **文本標籤**：看到任何文字、標誌、標牌都標記為有文本
2. **場景類型**：可以選擇多個類型，如道路+車輛
3. **一致性**：保持標註標準的一致性

### 數據集分割建議
處理完成後，建議按以下比例分割：
- 訓練集：80% (8,000張)
- 驗證集：10% (1,000張)
- 測試集：10% (1,000張)

確保同一場景/視頻的圖像不跨越不同數據集。

## 📞 技術支持

如遇到問題，請檢查：
1. 虛擬環境是否正確激活
2. 所有依賴是否正確安裝
3. 圖像文件夾路徑是否正確
4. 輸出目錄是否有寫入權限

運行環境測試：
```bash
python -c "import cv2, pandas, numpy; print('所有依賴正常')"
```
