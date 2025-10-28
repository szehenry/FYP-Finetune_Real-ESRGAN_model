# 圖像篩選工具環境設置指南

## 為什麼使用虛擬環境？

1. **避免依賴衝突** - 不同項目的包版本可能不兼容
2. **保持系統乾淨** - 避免污染系統Python環境
3. **便於管理** - 可以輕鬆安裝/卸載特定版本的包
4. **項目隔離** - 每個項目有獨立的依賴環境

## 設置步驟

### 1. 創建虛擬環境
```bash
# 進入項目目錄
cd "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP"

# 創建虛擬環境
python -m venv image_curator_env

# 或者使用conda（如果您使用conda）
conda create -n image_curator python=3.9
```

### 2. 激活虛擬環境
```bash
# macOS/Linux
source image_curator_env/bin/activate

# 或者使用conda
conda activate image_curator
```

### 3. 安裝依賴包
```bash
# 基礎依賴（必需）
pip install opencv-python pandas numpy pathlib2

# OCR功能（可選，但推薦）
pip install paddleocr

# 如果需要其他圖像處理功能
pip install pillow
```

### 4. 創建requirements.txt
```bash
# 生成依賴列表
pip freeze > requirements.txt
```

### 5. 運行工具
```bash
# 確保虛擬環境已激活
python image_curator.py

# 或帶OCR功能
python image_curator.py --ocr
```

### 6. 退出虛擬環境
```bash
# 完成工作後退出
deactivate

# 或者conda
conda deactivate
```

## 快速設置腳本

創建一個設置腳本來自動化這個過程：

```bash
#!/bin/bash
# setup.sh

echo "設置圖像篩選工具環境..."

# 創建虛擬環境
python -m venv image_curator_env

# 激活虛擬環境
source image_curator_env/bin/activate

# 升級pip
pip install --upgrade pip

# 安裝依賴
pip install opencv-python pandas numpy pathlib2 pillow

# 詢問是否安裝OCR
read -p "是否安裝OCR功能？(y/n): " install_ocr
if [[ $install_ocr == "y" || $install_ocr == "Y" ]]; then
    pip install paddleocr
    echo "OCR功能已安裝"
fi

# 生成requirements.txt
pip freeze > requirements.txt

echo "環境設置完成！"
echo "使用方法："
echo "1. 激活環境: source image_curator_env/bin/activate"
echo "2. 運行工具: python image_curator.py"
echo "3. 退出環境: deactivate"
```

## 故障排除

### 常見問題

1. **OpenCV安裝問題**
```bash
# 如果opencv-python安裝失敗，嘗試：
pip install opencv-python-headless
```

2. **PaddleOCR安裝問題**
```bash
# 如果PaddleOCR安裝失敗，可能需要：
pip install --upgrade setuptools wheel
pip install paddleocr
```

3. **權限問題**
```bash
# 如果遇到權限問題：
pip install --user opencv-python pandas numpy
```

## 項目結構建議

```
FYP/
├── image_curator_env/          # 虛擬環境
├── Images/                     # 圖像文件夾
├── image_curator.py           # 主程序
├── accepted_set.csv           # 輸出結果
├── requirements.txt           # 依賴列表
├── setup_environment.md       # 這個文件
└── README_image_curator.md    # 使用說明
```
