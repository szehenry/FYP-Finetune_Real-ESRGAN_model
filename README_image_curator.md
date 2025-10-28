# 图像筛选和标注工具使用说明

## 功能概述

这个工具专为Real-ESRGAN项目的数据集准备而设计，可以帮助您：
- 递归扫描文件夹中的图像
- 交互式查看和筛选图像质量
- 标注图像的文本内容和场景类型
- 自动保存标注结果到CSV文件
- 支持断点续传功能
- 可选的OCR自动文本检测

## 安装依赖

### 基础依赖（必需）
```bash
pip install opencv-python pandas numpy pathlib
```

### OCR功能（可选）
```bash
pip install paddleocr
```

## 使用方法

### 基本使用
```bash
python image_curator.py
```

### 自定义参数
```bash
# 指定图像文件夹和输出CSV路径
python image_curator.py --root /path/to/images --csv /path/to/output.csv

# 启用OCR自动文本检测
python image_curator.py --ocr

# 查看帮助
python image_curator.py --help
```

### 默认路径
- 图像文件夹：`/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/Images`
- 输出CSV：`/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP/accepted_set.csv`

## 操作说明

### 键盘控制
| 按键 | 功能 | 说明 |
|------|------|------|
| `a` | 接受图像 | 标记为接受并自动保存到CSV |
| `r` | 拒绝/跳过 | 跳过当前图像，不保存 |
| `t` | 切换文本标签 | 切换has_text标签（是/否） |
| `1-7` | 场景类型 | 选择场景类型（可多选） |
| `n` | 下一张 | 浏览下一张图像 |
| `b` | 上一张 | 浏览上一张图像 |
| `s` | 保存 | 手动保存当前标签（需先按a接受） |
| `h` | 帮助 | 显示控制说明 |
| `q` | 退出 | 退出程序 |

### 场景类型编码
| 数字键 | 场景类型 | 说明 |
|--------|----------|------|
| `1` | road | 道路场景 |
| `2` | building | 建筑物 |
| `3` | infrastructure | 基础设施 |
| `4` | vehicle | 车辆 |
| `5` | vegetation | 植被 |
| `6` | low-light | 低光照 |
| `7` | other | 其他 |

## 筛选标准

### 接受的图像特征
- **清晰度**：放大到100%时边缘清晰，无模糊
- **低噪声**：无明显颗粒感或条带
- **清晰视角**：物体可以用肉眼识别

### 拒绝的图像特征
- 严重模糊
- 噪声过重
- 压缩伪影明显

## 输出格式

CSV文件包含以下列：
- `filepath`：图像文件路径
- `has_text`：是否包含文本（0/1）
- `scene_type`：场景类型（逗号分隔的多个类型）
- `timestamp`：标注时间戳

示例：
```csv
filepath,has_text,scene_type,timestamp
/path/to/image1.jpg,1,"road,vehicle",2024-10-24T10:30:00
/path/to/image2.jpg,0,building,2024-10-24T10:31:00
```

## 断点续传功能

- 程序会自动检测已处理的图像
- 重新运行时从上次停止的位置继续
- 每次接受图像时自动保存，避免数据丢失

## OCR功能

启用OCR后：
- 自动检测图像中的文本
- 高置信度文本会自动设置has_text=True
- 您仍可以手动切换标签

## 数据集分割建议

处理完成后，建议按以下方式分割数据集：
- 训练集：8,000张图像
- 验证集：1,000张图像  
- 测试集：1,000张图像

确保同一场景/视频的图像不会跨越不同的数据集分割。

## 故障排除

### 常见问题
1. **图像无法显示**：检查图像文件是否损坏或格式不支持
2. **OCR功能不可用**：安装PaddleOCR：`pip install paddleocr`
3. **CSV保存失败**：检查输出路径是否有写入权限

### 支持的图像格式
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

## 性能优化建议

1. 对于大量图像，建议分批处理
2. 启用OCR会增加处理时间，但提高标注准确性
3. 定期备份CSV文件以防数据丢失

## 快速开始示例

```bash
# 1. 进入项目目录
cd "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP"

# 2. 运行工具（基础版本）
python image_curator.py

# 3. 运行工具（带OCR）
python image_curator.py --ocr

# 4. 查看结果
head -10 accepted_set.csv
```

处理完成后，您将获得一个包含高质量图像标注的CSV文件，可用于后续的数据集分割和模型训练。
