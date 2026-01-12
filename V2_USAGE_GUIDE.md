# evaluate_realesrgan_only_v2.py 使用指南

## 🎯 新功能

### ✅ 已添加功能

1. **自动恢复 V1 进度** 
   - 自动检测现有的 `realesrgan_results.csv`
   - 跳过已处理的 3,757 张图像
   - 从剩余的 6,933 张继续

2. **保存增强图像**
   - 从现在开始保存所有增强后的图像
   - 保存位置：`D:\baseline_results_realesrgan_only\enhanced_images\`
   - 格式：PNG（高质量，无损）
   - 命名：`enhanced_[原文件名].png`

3. **Checkpoint 恢复**
   - 每 100 张自动保存进度
   - 随时 Ctrl+C 暂停
   - 重新运行自动继续

4. **强化错误处理**
   - OOM 时跳过图像并继续
   - 更强的内存清理

5. **优化配置**
   - TILE = 512（更安全，减少 OOM）
   - FP32（更稳定）

---

## 📊 你的问题解答

### Q1: V2 会继续 V1 的进度吗？

**✅ 是的！**

```python
V2 会自动检测并恢复：
1. 首先查找 checkpoint.json（如果存在）
2. 如果没有，查找 realesrgan_results.csv（你的 V1 结果）
3. 读取已处理的图像列表（3,757 张）
4. 跳过这些图像
5. 从剩余的 6,933 张继续

输出：
🔄 檢測到現有結果文件，嘗試恢復...
✅ 已從現有結果恢復: 已處理 3,757 張
💾 增強圖像將開始保存（從現在開始）
```

### Q2: 为什么内存还会满？（代码有清理）

**原因分析：**

```python
虽然代码每张图后都清理，但：

1. TILE=800 对超大图像不够安全
   例：4000×3000 图像
   - 需要 20 tiles (5×4)
   - 每个 tile: 800×800 输入 → 3200×3200 输出
   - 单个 tile 峰值内存：2-3GB
   - 总峰值：可能超过 6GB（RTX 3060）

2. FP16 内存碎片
   - 虽然省内存，但长时间运行会碎片化
   - 碎片无法被清理回收

3. LPIPS 额外开销
   - 加载感知网络：~1GB
   - 700 张随机抽样时需要计算

解决方案：V2 使用 TILE=512 + FP32
```

### Q3: 降低 TILE 会更慢吗？

**✅ 是的，但更安全！**

```python
速度对比：

TILE=800 (你之前的配置):
- 3000×2000 图像 → 12 tiles (4×3)
- 处理时间：~65 秒/张
- OOM 风险：高 ❌

TILE=512 (V2 配置):
- 3000×2000 图像 → 30 tiles (6×5)
- 处理时间：~100 秒/张 (1.5倍慢)
- OOM 风险：很低 ✅

原因：tiles 数量 ∝ (图像尺寸/TILE)²
     512 vs 800 → (800/512)² = 2.4倍 tiles
     但实际慢 1.5倍（因为 GPU 利用率更高）
```

### Q4: 时间估算

**基于你的数据：**

```python
已完成：
- 图像：3,757 张
- 时间：68.4 小时 (2.85 天)
- 速度：65.4 秒/张 (TILE=800, FP16)
- 失败：~700 张 OOM

剩余估算：

选项 A：继续 TILE=800 (不推荐)
- 剩余：6,933 张
- 时间：6,933 × 65.4 秒 = 125.8 小时 (5.2 天)
- ❌ 但会继续 OOM，可能 30% 失败
- ❌ 最终成功率：~70%

选项 B：使用 V2 TILE=512 (推荐) ⭐
- 剩余：6,933 张
- 速度：~100 秒/张 (更慢但安全)
- 时间：6,933 × 100 秒 = 192.6 小时 (8 天)
- ✅ OOM 风险：<5%
- ✅ 最终成功率：~95%

总时间（从开始到现在）：
- 已用：2.85 天
- 还需：8 天
- 总计：~10.85 天

实际可能更快：
- 小图像处理更快（不需要 30 tiles）
- 平均可能 ~7-8 天完成
```

### Q5: FP16 是什么？

```python
FP16 vs FP32 对比：

┌─────────────┬───────────┬───────────┐
│ 特性        │ FP16      │ FP32      │
├─────────────┼───────────┼───────────┤
│ 位数        │ 16 位     │ 32 位     │
│ 内存占用    │ 一半      │ 标准      │
│ 速度        │ 更快*     │ 标准      │
│ 稳定性      │ 较差      │ 很好 ✅   │
│ 精度        │ 低        │ 标准      │
│ 内存碎片    │ 容易产生  │ 较少      │
└─────────────┴───────────┴───────────┘

* 需要 GPU 支持 Tensor Cores (RTX 30系列支持)

FP16 优势：
✅ 内存节省 ~30-40%
✅ 理论上快 20-30%（如果支持）

FP16 问题（你遇到的）：
❌ 内存碎片化（长时间运行）
❌ 数值不稳定
❌ 某些操作可能 OOM

推荐：FP32（V2 默认）
```

### Q6: 增强图像保存

```python
V2 配置：

SAVE_ENHANCED_IMAGES = True  # 启用保存
ENHANCED_IMAGE_FORMAT = "png"  # PNG 格式

保存位置：
D:\baseline_results_realesrgan_only\enhanced_images\

文件命名：
原文件：0000001_00012_d_0000001_global_blur.jpg
保存为：enhanced_0000001_00012_d_0000001_global_blur.png

特点：
✅ 只保存新处理的图像（从现在开始）
✅ 已处理的 3,757 张不会重新生成
✅ PNG 格式（无损，质量最好）
✅ 压缩级别 3（平衡质量和大小）

如果想保存为 JPG（节省空间）：
修改：ENHANCED_IMAGE_FORMAT = "jpg"
大小差异：PNG ~3MB，JPG ~500KB
```

---

## 🚀 立即使用

### 步骤 1：停止旧版本

```powershell
# 在运行 V1 的终端按 Ctrl+C
Ctrl + C

# 确认停止
```

### 步骤 2：运行 V2

```powershell
# 激活虚拟环境
venv_baseline\Scripts\activate

# 运行 V2
python evaluate_realesrgan_only_v2.py

# 你会看到：
🔄 檢測到現有結果文件，嘗試恢復...
✅ 已從現有結果恢復: 已處理 3,757 張
💾 增強圖像將開始保存（從現在開始）

💡 評估策略:
  ✅ PSNR/SSIM: 所有 10,690 張圖像
  ✅ LPIPS: 隨機抽樣 700 張
  🔧 記憶體管理: 每張圖像後清理 GPU 快取
  💾 Checkpoint: 每 100 張保存
  📸 增強圖像: 啟用保存 (png)
  ⚡ 使用設備: cuda
  🎛️  TILE: 512 (FP32 穩定模式)

📈 預計時間: ~7-8 天（剩餘 6,933 張）
```

### 步骤 3：监控进度

```powershell
# 进度条显示
Real-ESRGAN 評估:  35%|████████ | 3757/10690 [00:15:23<00:45:12]

# 定期输出
💾 Checkpoint 已保存 (成功: 3857, 失敗: 5)

# 检查增强图像
ls D:\baseline_results_realesrgan_only\enhanced_images\
```

### 步骤 4：随时暂停/继续

```powershell
# 暂停（任何时候）
Ctrl + C

# 继续（重新运行）
python evaluate_realesrgan_only_v2.py

# 自动从上次位置继续！
✅ 已載入 Checkpoint: 已處理 4,500 張
```

---

## ⚙️ 配置调整

### 如果仍然 OOM

```python
# 编辑 evaluate_realesrgan_only_v2.py
# Line 103-105

TILE = 400  # 从 512 降到 400（更安全但更慢）
FP32 = True  # 保持 FP32
```

### 如果想更快（但有风险）

```python
TILE = 800  # 提高到 800（更快但可能 OOM）
FP32 = True  # 保持 FP32
```

### 如果不想保存增强图像（节省空间）

```python
SAVE_ENHANCED_IMAGES = False  # 不保存增强图像
```

### 如果想保存为 JPG（节省空间）

```python
SAVE_ENHANCED_IMAGES = True
ENHANCED_IMAGE_FORMAT = "jpg"  # 改为 jpg

# 空间节省：
# PNG: ~3MB/张 × 6,933 = ~20GB
# JPG: ~500KB/张 × 6,933 = ~3.5GB
```

---

## 📈 预期结果

### 完成后你会有

```
D:\baseline_results_realesrgan_only\
├── checkpoint.json  # Checkpoint 数据
├── enhanced_images/  # 增强图像（6,933 张）
│   ├── enhanced_0000xxx_xxx.png
│   └── ...
├── leaderboards/
│   └── realesrgan_results.csv  # 完整结果（10,690 张）
└── ...
```

### 统计信息

```python
预期成功率：
- TILE=512 + FP32: ~95-98%
- 失败：~100-200 张（超大图像或损坏文件）

最终结果：
- 成功：~10,500 张
- 失败：~200 张
- 总时间：~10-11 天（从开始到结束）
```

---

## 🆘 故障排除

### 问题：还是 OOM

```python
解决：降低 TILE
TILE = 400  # 或者 300
```

### 问题：速度太慢

```python
考虑：云服务器
- Vast.ai RTX 4000: $0.2/小时
- TILE=1200
- 15-20 小时完成（~$3-4）
```

### 问题：磁盘空间不够

```python
解决：
1. SAVE_ENHANCED_IMAGES = False  # 不保存
2. ENHANCED_IMAGE_FORMAT = "jpg"  # 用 JPG
3. 只保存部分：修改代码添加条件判断
```

### 问题：想重新开始

```python
# 删除 checkpoint
del D:\baseline_results_realesrgan_only\checkpoint.json

# 删除结果（如果想完全重跑）
del D:\baseline_results_realesrgan_only\leaderboards\realesrgan_results.csv

# 重新运行
python evaluate_realesrgan_only_v2.py
```

---

## 💡 推荐配置总结

### 本地运行（RTX 3060）⭐

```python
TILE = 512
FP32 = True
SAVE_ENHANCED_IMAGES = True
ENHANCED_IMAGE_FORMAT = "png"  # 或 "jpg" 节省空间

预计时间：7-8 天（剩余图像）
成功率：95%+
成本：免费
```

### 云服务器（更快）⭐⭐⭐

```python
TILE = 1200  # 更大
FP32 = True
SAVE_ENHANCED_IMAGES = True

GPU: RTX 4000/T4
预计时间：15-20 小时（全部图像）
成功率：98%+
成本：$15-25
```

---

## ✅ 开始吧！

```powershell
# 1. 停止旧版本
Ctrl + C

# 2. 运行新版本
python evaluate_realesrgan_only_v2.py

# 3. 让它运行 7-8 天

# 4. 完成！
```

**V2 会自动：**
- ✅ 恢复你的 3,757 张进度
- ✅ 处理剩余 6,933 张
- ✅ 保存所有新的增强图像
- ✅ 定期保存 checkpoint
- ✅ 处理 OOM 错误
- ✅ 生成完整结果

祝顺利！🚀

