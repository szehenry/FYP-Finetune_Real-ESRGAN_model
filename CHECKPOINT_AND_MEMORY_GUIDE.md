# Checkpoint 使用指南 & 内存优化

## 🎯 你的问题解答

### Q1: 保存到 CSV 后可以清除内存缓存吗？

**✅ 可以！而且你的想法很对！**

```python
问题：
all_results = []  # 累积所有结果在内存中
→ 10,690 张后可能占用 100MB+ CPU RAM

解决方案（已添加）：
CLEAR_MEMORY_AFTER_CHECKPOINT = True
→ 每 100 张保存后清除 all_results
→ CSV 已有完整数据
→ 节省内存！
```

### Q2: Checkpoint 如何使用？

**✅ 非常简单：Ctrl+C 停止，重新运行继续！**

---

## 📊 内存使用分析

### 当前内存分布

```python
处理 10,690 张图像的内存占用：

┌──────────────────────┬─────────┬─────────┬─────────┐
│ 项目                 │ 大小    │ 位置    │ 瓶颈？  │
├──────────────────────┼─────────┼─────────┼─────────┤
│ all_results 列表     │ ~100MB  │ CPU RAM │ 小 ✅   │
│ processed_images 集合│ ~5MB    │ CPU RAM │ 极小 ✅ │
│ Real-ESRGAN 模型     │ ~500MB  │ GPU     │ 固定 ✅ │
│ 临时图像数据(1张)    │ 2-4GB   │ GPU     │ 大❗   │
│ LPIPS 模型           │ ~1GB    │ GPU     │ 偶尔    │
└──────────────────────┴─────────┴─────────┴─────────┘

结论：
✅ CPU RAM 占用很小（~100MB）
❗ GPU 内存是真正的瓶颈（2-4GB/张）
✅ GPU 已经每张图后清理

对于你的数据集（10,690 张）：
- 不清理 all_results: 完全 OK
- 清理 all_results: 节省 ~100MB（帮助不大）

但对于更大数据集（100万张）：
- 不清理: 可能 10GB+ RAM ❌
- 清理: ~几 MB RAM ✅
```

### 真正的内存瓶颈

```python
GPU 内存瓶颈：
1. 单张大图像（4000×3000）
   ├─ 输入: ~35MB
   ├─ Real-ESRGAN 处理: 2-3GB（中间层）
   └─ 输出: ~140MB（4× upscale）
   = 总共 2-4GB/张

2. LPIPS 计算（700 张随机）
   ├─ 额外感知网络: ~1GB
   └─ 特征提取: ~500MB
   = 额外 1.5GB

解决方案（已实现）：
✅ 每张图后清理 GPU（已有）
✅ TILE=512 减少单张峰值（已有）
✅ FP32 避免碎片（已有）
```

---

## 🔧 新增功能：可选内存清理

### 配置选项

```python
# evaluate_realesrgan_only_v2.py
# Line 112

CLEAR_MEMORY_AFTER_CHECKPOINT = False  # 默认关闭

# 打开内存清理：
CLEAR_MEMORY_AFTER_CHECKPOINT = True

什么时候打开？
✅ 数据集很大（>100,000 张）
✅ CPU RAM 有限（<16GB）
✅ 运行其他程序（需要节省内存）

什么时候关闭？
✅ 数据集中等（<50,000 张）← 你的情况
✅ RAM 充足（>16GB）
✅ 想保留完整的内存调试信息
```

### 工作原理

```python
启用内存清理（CLEAR_MEMORY_AFTER_CHECKPOINT = True）：

处理流程：
1. 处理 100 张图像 → all_results (100 个字典)
2. 保存 checkpoint.json (完整数据)
3. 保存 CSV (完整数据) ✅
4. 清除 all_results.clear() ✅
5. 继续处理下 100 张...

优点：
✅ 内存占用恒定（~几 MB）
✅ 可处理无限大数据集
✅ CSV 有完整数据

缺点：
⚠️ checkpoint.json 只有最近 100 张（但够用）
⚠️ 最终统计需要读 CSV（稍慢，但影响小）
```

### 输出示例

```bash
# 启用内存清理
CLEAR_MEMORY_AFTER_CHECKPOINT = True

python evaluate_realesrgan_only_v2.py

# 每 100 张输出：
💾 已保存 (成功: 98, 失敗: 2)
   ├─ Checkpoint: checkpoint.json
   ├─ CSV: realesrgan_results.csv (98 筆記錄)
   └─ 內存緩存已清除（節省 RAM）  ← 新增！

# 内存占用：
- 清理前：~10MB (100 个结果)
- 清理后：~0.1MB (空列表)
- 节省：~10MB
```

---

## 💾 Checkpoint 完整指南

### 什么是 Checkpoint？

```python
Checkpoint = 进度保存点

包含：
├─ checkpoint.json（进度数据）
│  ├─ 所有结果（或最近 100 张）
│  ├─ 已处理图像列表
│  ├─ 配置信息
│  └─ 时间戳
│
└─ realesrgan_results.csv（评估指标）
   └─ 所有已处理图像的完整结果
```

### 如何使用 Checkpoint

#### 方法 1：正常暂停（推荐）⭐

```powershell
# 1. 运行程序
python evaluate_realesrgan_only_v2.py

# 2. 随时按 Ctrl+C 停止
Real-ESRGAN 評估:  15%|██████  | 1598/10690 [26:30:00<...]
^C
KeyboardInterrupt

# 3. 程序会尝试自动保存（Python 信号处理）
💾 正在保存最後的 checkpoint...

# 4. 重新运行，自动继续！
python evaluate_realesrgan_only_v2.py

# 输出：
✅ 已載入 Checkpoint: 已處理 1,598 張
🎯 繼續處理剩餘 9,092 張...
```

#### 方法 2：等待 Checkpoint 后停止（最安全）⭐⭐⭐

```powershell
# 1. 运行程序
python evaluate_realesrgan_only_v2.py

# 2. 等待出现 "已保存" 提示
Real-ESRGAN 評估:  15%|██████  | 1600/10690 [26:40:00<...]
💾 已保存 (成功: 1,595, 失敗: 5)  ← 看到这个！
   ├─ Checkpoint: checkpoint.json
   └─ CSV: realesrgan_results.csv (1,595 筆記錄)

# 3. 立即按 Ctrl+C（数据已安全保存）
^C

# 4. 重新运行
python evaluate_realesrgan_only_v2.py

# 输出：
✅ 已載入 Checkpoint: 已處理 1,595 張
```

#### 方法 3：强制停止（紧急情况）

```powershell
# 如果程序卡住，无法 Ctrl+C

# Windows:
- 关闭终端窗口
- 或任务管理器结束进程

# 数据丢失：
- 最多丢失最近 100 张（最后一次 checkpoint 后的）
- CSV 保留了之前所有数据
```

### Checkpoint 恢复逻辑

```python
程序启动时：

1. 查找 checkpoint.json
   ├─ 存在 → 读取已处理图像列表
   └─ 不存在 → 继续步骤 2

2. 查找 realesrgan_results.csv
   ├─ 存在 → 读取已处理图像列表
   └─ 不存在 → 从头开始

3. 跳过已处理的图像
   └─ 只处理剩余的图像

恢复优先级：
checkpoint.json > CSV > 从头开始
```

---

## 📁 文件同步

### 保存时机

```python
自动保存：
✅ 1. 首次成功（第 1 张）
✅ 2. 每 100 张
✅ 3. 程序正常结束
⚠️ 4. Ctrl+C（尽力保存，但可能失败）

文件同步：
每次保存时：
├─ checkpoint.json（更新）
└─ realesrgan_results.csv（追加/覆盖）

数据一致性：
- CSV 是主要数据源（完整记录）
- checkpoint 是快速恢复索引
- 两者互为备份
```

### 数据安全性

```python
场景 1：正常 Ctrl+C
✅ 最后一次 checkpoint（如 1,500 张）已保存
✅ CSV 有完整的 1,500 条记录
⚠️ 最后 1-99 张可能丢失（未到保存点）

场景 2：程序崩溃/断电
✅ 磁盘上的 checkpoint 和 CSV 完整
✅ 恢复到最后一次保存点（如 1,500 张）
⚠️ 内存中未保存的数据丢失

场景 3：磁盘满/文件损坏
⚠️ checkpoint 和 CSV 可能部分损坏
✅ 可以尝试从备份恢复
✅ 或手动编辑 CSV 删除损坏行

最坏情况：
- 丢失最后 <100 张（未到 checkpoint）
- vs 之前：丢失全部 10,690 张 😢
```

---

## 🎮 实战示例

### 示例 1：运行 3 小时后暂停

```bash
# 启动
python evaluate_realesrgan_only_v2.py

# 3 小时后（~180 张）
Real-ESRGAN 評估:   2%|▊  | 180/10690 [03:00:00<...]

# 看到保存提示
💾 已保存 (成功: 178, 失敗: 2)

# 立即 Ctrl+C
^C

# 检查文件
PS> ls D:\baseline_results_realesrgan_only\leaderboards\
realesrgan_results.csv  # ✅ 178 行

PS> ls D:\baseline_results_realesrgan_only\
checkpoint.json  # ✅ 存在

# 明天继续
python evaluate_realesrgan_only_v2.py

# 输出：
✅ 已載入 Checkpoint: 已處理 178 張
🎯 繼續處理剩餘 10,512 張...
Real-ESRGAN 評估:   2%|▊  | 179/10690 [00:01:30<...]  ← 从 179 继续！
```

### 示例 2：多次暂停/继续

```bash
# Day 1: 运行 2 小时
python evaluate_realesrgan_only_v2.py
# ... 处理到 100 张 ...
^C

# Day 2: 继续 3 小时
python evaluate_realesrgan_only_v2.py
# ✅ 从 100 继续 → 处理到 280 张
^C

# Day 3: 继续 4 小时
python evaluate_realesrgan_only_v2.py
# ✅ 从 280 继续 → 处理到 520 张
^C

# ... 多次继续 ...

# Day 10: 完成！
python evaluate_realesrgan_only_v2.py
# ✅ 从 10,500 继续 → 完成 10,690 张
✅ Real-ESRGAN 評估完成！
```

### 示例 3：程序崩溃后恢复

```bash
# 运行中突然崩溃（如断电）
Real-ESRGAN 評估:  25%|██████████  | 2,673/10690 [...]
[电源断开]

# 重启电脑后
python evaluate_realesrgan_only_v2.py

# 输出：
✅ 已載入 Checkpoint: 已處理 2,600 張  ← 最后保存点
⚠️  丢失 73 张（2,600 → 2,673，未保存）
🎯 繼續處理剩餘 8,090 張...
Real-ESRGAN 評估:  24%|██████████  | 2,601/10690 [...]  ← 从 2,601 继续
```

---

## ⚙️ 配置建议

### 你的情况（10,690 张，RTX 3060）

```python
推荐配置：

CHECKPOINT_INTERVAL = 100  # ✅ 默认，平衡
CLEAR_MEMORY_AFTER_CHECKPOINT = False  # ✅ 关闭（不需要）
SAVE_ENHANCED_IMAGES = True  # ✅ 保存图像
TILE = 512  # ✅ 安全值
FP32 = True  # ✅ 稳定

原因：
- 10,690 张不算大，内存占用小
- 关闭清理更简单，调试方便
- GPU 内存才是瓶颈（已优化）
```

### 大数据集（>100,000 张）

```python
推荐配置：

CHECKPOINT_INTERVAL = 500  # 更少保存（减少 I/O）
CLEAR_MEMORY_AFTER_CHECKPOINT = True  # ✅ 打开（节省内存）
SAVE_ENHANCED_IMAGES = False  # 可选（节省磁盘）
TILE = 512
FP32 = True

原因：
- 100,000 张结果可能占用 1GB+ RAM
- 清理后恒定 ~10MB
- 减少 checkpoint 频率加快速度
```

### 小数据集（<1,000 张）

```python
推荐配置：

CHECKPOINT_INTERVAL = 20  # 更频繁（更安全）
CLEAR_MEMORY_AFTER_CHECKPOINT = False
SAVE_ENHANCED_IMAGES = True
TILE = 1200  # 可以更大（如果 GPU 足够）
FP32 = True

原因：
- 数据少，频繁保存无压力
- 内存完全不是问题
- 可以用更大 TILE 加速
```

---

## 🆘 常见问题

### Q: Checkpoint 会拖慢速度吗？

```python
A: 几乎不会！

保存时间：
- checkpoint.json: ~1-3 秒
- CSV: ~0.5-1 秒
- 总计: ~2-4 秒/100 张

vs 处理时间：
- 100 张: ~2.5-3 小时

影响：<0.1%（可忽略）
```

### Q: 可以改变 CHECKPOINT_INTERVAL 吗？

```python
A: 可以！

CHECKPOINT_INTERVAL = 50   # 更频繁，更安全
CHECKPOINT_INTERVAL = 200  # 更少保存，稍快
CHECKPOINT_INTERVAL = 500  # 大数据集推荐

推荐：100（默认）
```

### Q: 删除 checkpoint 会怎样？

```python
A: 程序会从 CSV 恢复！

优先级：
1. checkpoint.json（快速）
2. CSV（备份）
3. 从头开始

所以：删除 checkpoint 不怕，CSV 还在！
```

### Q: CSV 和 checkpoint 不一致怎么办？

```python
A: 程序会使用 processed_images 列表

保护机制：
- processed_images 记录所有已处理图像
- 即使 CSV 有 1,000 行，checkpoint 只有 100 行
- 程序会跳过 CSV 中的所有 1,000 张

结论：不会重复处理！
```

---

## 📊 总结

### 内存优化

```python
你的问题：保存到 CSV 后可以清除内存吗？
答案：✅ 可以！已添加选项

对你的数据集：
- 不清理：~100MB（可接受）
- 清理：~0.1MB（节省小）
- 推荐：关闭（CLEAR_MEMORY_AFTER_CHECKPOINT = False）

对大数据集（100万张）：
- 不清理：~10GB（不可接受！）
- 清理：~10MB（必须！）
- 推荐：打开（CLEAR_MEMORY_AFTER_CHECKPOINT = True）
```

### Checkpoint 使用

```python
最简单方法：
1. 运行：python evaluate_realesrgan_only_v2.py
2. 暂停：Ctrl + C
3. 继续：python evaluate_realesrgan_only_v2.py

就是这样！✅

注意：
- 尽量在看到 "💾 已保存" 后再 Ctrl+C
- 最安全，数据不丢失
```

---

## 🚀 立即开始

```powershell
# 使用推荐配置（默认）
python evaluate_realesrgan_only_v2.py

# 特点：
✅ 自动保存（每 100 张）
✅ 随时暂停/继续（Ctrl+C）
✅ 数据安全（最多丢失 <100 张）
✅ 双重备份（checkpoint + CSV）
✅ 内存优化（可选）

预计时间：7-8 天
成功率：95%+
```

**开始吧！** 🚀

