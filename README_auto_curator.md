# 自动图像筛选工具 (Auto Image Curator)

这是一个为Real-ESRGAN项目设计的自动化图像筛选和标注工具，能够自动识别高质量图像、检测对象和场景类型，并将结果保存到CSV文件中。

## 主要功能

### 🔍 自动质量检测
- **运动模糊检测**：使用拉普拉斯算子和Sobel梯度检测图像模糊程度
- **光照条件分析**：识别低光照、过曝和夜间场景
- **阴影细节评估**：判断阴影区域是否保留足够细节
- **综合质量评分**：基于多个指标的综合评估

### 🎯 智能对象识别
- **YOLO检测**：使用YOLOv8进行高精度对象检测
- **OpenCV检测**：人脸和车辆检测作为补充
- **场景特征分析**：基于颜色和边缘特征识别道路、建筑、植被等

### 📝 自动标注功能
- **文本检测**：使用PaddleOCR检测图像中的文字
- **场景分类**：自动识别道路、建筑、车辆、人物、植被等
- **时间判断**：区分白天和夜间场景
- **质量等级**：高质量、中等质量分级

### 📊 数据管理
- **CSV输出**：与原始工具兼容的CSV格式
- **拒绝图像处理**：自动移动低质量图像到单独文件夹
- **统计报告**：详细的处理统计和分析结果

## 安装依赖

### 基础安装（必需）
```bash
pip install opencv-python pandas numpy
```

### 完整功能安装（推荐）
```bash
pip install -r requirements_auto_curator.txt
```

### 分步安装
```bash
# 基础图像处理
pip install opencv-python>=4.8.0 pandas>=2.0.0 numpy>=1.24.0

# YOLO对象检测（推荐）
pip install ultralytics>=8.0.0 torch torchvision

# OCR文本检测（可选）
pip install paddleocr>=2.7.0

# 图像质量增强（可选）
pip install scikit-image pillow
```

## 使用方法

### 基本用法
```bash
python auto_image_curator.py --root /path/to/your/images
```

### 完整参数
```bash
python auto_image_curator.py \
    --root /path/to/your/images \
    --csv /path/to/output.csv \
    --rejected /path/to/rejected/folder \
    --quality-threshold 0.7 \
    --no-ocr
```

### 参数说明
- `--root`：图像文件夹路径（必需）
- `--csv`：CSV输出文件路径（可选，默认在输入文件夹下）
- `--rejected`：拒绝图像文件夹路径（可选，默认在输入文件夹下）
- `--quality-threshold`：质量阈值 0.0-1.0（默认0.6）
- `--no-ocr`：禁用OCR文本检测

## 输出格式

### CSV文件列
- `filepath`：图像文件路径
- `has_text`：是否包含文本 (0/1)
- `scene_type`：场景类型（逗号分隔）
- `quality_level`：质量等级 (high/medium)
- `is_night`：是否夜间 (0/1)
- `quality_score`：质量分数 (0.0-1.0)
- `blur_score`：模糊分数
- `mean_brightness`：平均亮度
- `shadow_detail_visible`：阴影细节可见 (0/1)
- `timestamp`：处理时间戳

### 场景类型
- `road`：道路、街道
- `building`：建筑物
- `infrastructure`：基础设施
- `vehicle`：车辆
- `vegetation`：植被
- `person_animal`：人物/动物
- `other`：其他

## 质量标准

### 接受标准
✅ **高质量图像**
- 无明显运动模糊
- 光照条件良好
- 阴影区域细节可见
- 综合质量分数 ≥ 阈值

✅ **可接受的夜间图像**
- 虽然整体较暗但细节可观察
- 没有过度噪点
- 关键区域有足够对比度

### 拒绝标准
❌ **低质量图像**
- 严重运动模糊
- 过度曝光或欠曝
- 阴影区域完全黑暗
- 整体质量分数过低

## 使用示例

### 示例1：处理单个文件夹
```bash
# 处理Images文件夹，输出到auto_accepted_set.csv
python auto_image_curator.py --root ./Images

# 结果：
# - ./Images/auto_accepted_set.csv（接受的图像信息）
# - ./Images/rejected_images/（拒绝的图像）
```

### 示例2：自定义输出路径
```bash
# 自定义CSV和拒绝文件夹路径
python auto_image_curator.py \
    --root /Users/henrysze/Pictures/Dataset1 \
    --csv /Users/henrysze/Results/dataset1_results.csv \
    --rejected /Users/henrysze/Results/dataset1_rejected
```

### 示例3：调整质量标准
```bash
# 使用更严格的质量标准
python auto_image_curator.py \
    --root ./Images \
    --quality-threshold 0.8

# 使用更宽松的质量标准
python auto_image_curator.py \
    --root ./Images \
    --quality-threshold 0.4
```

## 测试工具

运行测试脚本验证功能：
```bash
python test_auto_curator.py
```

测试脚本会：
1. 创建各种类型的测试图像
2. 运行自动筛选流程
3. 显示检测结果和统计信息
4. 验证各个功能模块

## 性能优化建议

### 🚀 提高处理速度
1. **禁用不需要的功能**：
   ```bash
   # 如果不需要文本检测
   python auto_image_curator.py --root ./Images --no-ocr
   ```

2. **调整质量阈值**：
   - 较低阈值（0.4-0.5）：接受更多图像，处理更快
   - 较高阈值（0.7-0.8）：更严格筛选，质量更高

3. **批量处理**：
   ```bash
   # 处理多个文件夹
   for folder in Dataset1 Dataset2 Dataset3; do
       python auto_image_curator.py --root ./$folder
   done
   ```

### 🎯 提高检测准确性
1. **确保依赖完整**：安装所有推荐的依赖包
2. **调整阈值**：根据你的数据集特点调整质量阈值
3. **检查结果**：定期检查CSV输出和拒绝文件夹的结果

## 故障排除

### 常见问题

**Q: YOLO模型下载失败**
```bash
# 手动下载模型
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Q: OCR检测不工作**
```bash
# 检查PaddleOCR安装
pip install paddleocr paddlepaddle
```

**Q: 内存不足**
- 减少并发处理
- 使用较小的YOLO模型
- 禁用不必要的功能

**Q: 检测结果不准确**
- 调整质量阈值
- 检查图像质量
- 验证依赖包版本

### 日志和调试
程序会输出详细的处理信息：
- 每个图像的处理结果
- 检测到的对象和场景
- 质量评分和拒绝原因
- 最终统计信息

## 与原始工具的对比

| 功能 | 原始工具 | 自动工具 |
|------|----------|----------|
| 图像查看 | 手动逐张查看 | 自动批量处理 |
| 质量判断 | 人工判断 | 算法自动检测 |
| 对象识别 | 手动标注 | AI自动识别 |
| 文本检测 | 可选OCR | 自动OCR检测 |
| 处理速度 | 慢（手动） | 快（自动） |
| 一致性 | 依赖操作者 | 算法一致 |
| CSV格式 | ✅ 兼容 | ✅ 兼容 |

## 技术细节

### 模糊检测算法
- 拉普拉斯算子方差检测
- Sobel梯度分析
- 动态阈值调整（基于图像尺寸）

### 光照分析方法
- 亮度直方图分析
- 低光照比例计算
- 阴影区域纹理检测

### 对象检测流程
1. YOLOv8主要检测
2. OpenCV级联分类器补充
3. 基于颜色和边缘的场景分析
4. 结果融合和映射

## 更新日志

### v1.0.0 (2024-10-25)
- 初始版本发布
- 实现基础质量检测
- 集成YOLO和OCR
- 支持自动拒绝图像处理
- 兼容原始CSV格式

---

## 联系和支持

如有问题或建议，请检查：
1. 依赖包是否正确安装
2. 图像文件夹路径是否正确
3. 运行测试脚本验证功能
4. 查看程序输出的错误信息
