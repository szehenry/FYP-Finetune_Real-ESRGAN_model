# 合并对比样本指南

## 🎯 你的方案（更合理）

### 当前结构

```
现有结果：

D:\baseline_results_old\
└── leaderboards\
    ├── full_results_with_gt.csv  # 6个传统方法的完整结果
    ├── leaderboard_*.csv  # 各退化类型的排行榜
    ├── evaluation_summary.json
    └── comparison_samples\  # 8个子图（GT + Degraded + 6方法）

D:\baseline_results_realesrgan_only\
├── enhanced_images\  # Real-ESRGAN增强图（1,500张或10,690张）
├── sampled_images.txt  # 采样列表（如果启用采样）
└── leaderboards\
    └── realesrgan_results.csv  # Real-ESRGAN结果
```

### 你的建议（完美！）✅

```
合并策略：

1. Leaderboards：保持独立 ✅
   ├─ D:\baseline_results_old\leaderboards\
   │  └─ 传统方法的排行榜（独立）
   └─ D:\baseline_results_realesrgan_only\leaderboards\
      └─ Real-ESRGAN排行榜（独立）

2. Comparison samples：生成新的合并版本 ✅
   └─ D:\baseline_results_merged\comparison_samples\
      └─ 9个方法对比（GT + Degraded + 6传统 + Real-ESRGAN）

3. Enhanced images：只在Real-ESRGAN文件夹 ✅
   └─ D:\baseline_results_realesrgan_only\enhanced_images\
      └─ Real-ESRGAN增强图

优点：
✅ 结构清晰
✅ 各方法结果独立
✅ 不会弄乱原有结构
✅ 对比样本统一查看
✅ Enhanced images不重复
```

---

## 📊 文件结构对比

### 之前的方案（merge_baseline_results.py）

```
❌ 问题：合并所有东西

D:\baseline_results_merged\
├── leaderboards\
│   ├── full_results_merged.csv  # 合并的CSV
│   └── leaderboard_*.csv  # 合并的排行榜
├── comparison_samples\  # 9个方法对比
└── enhanced_images\  # Real-ESRGAN图像

缺点：
❌ 结构混乱
❌ 不好区分各方法结果
❌ Enhanced images重复
```

### 现在的方案（merge_comparison_samples_only.py）⭐

```
✅ 优点：只合并对比样本

保持原有结构：
D:\baseline_results_old\leaderboards\  # 传统方法（独立）
D:\baseline_results_realesrgan_only\  # Real-ESRGAN（独立）

新增合并输出：
D:\baseline_results_merged\
└── comparison_samples\  # 只有对比样本（9个方法）

优点：
✅ 原有结果不动
✅ 结构清晰
✅ 查看方便
```

---

## 🚀 使用方法

### 步骤 1：运行 Baseline 评估（已完成）

```powershell
# 你已经运行过了
python baseline_evaluation_with_gt-HenryCC.py

# 结果：
D:\baseline_results_old\leaderboards\
├── full_results_with_gt.csv  # 6个方法
└── comparison_samples\  # 8个子图
```

### 步骤 2：运行 Real-ESRGAN 评估

```powershell
# 使用采样（推荐）
python evaluate_realesrgan_only_v2.py

# 配置：
# ENABLE_SAMPLING = True
# SAMPLES_PER_TYPE = 500
# TILE = 800

# 预计：~24 小时

# 结果：
D:\baseline_results_realesrgan_only\
├── enhanced_images\  # 1,500 张增强图
└── leaderboards\
    └── realesrgan_results.csv
```

### 步骤 3：合并对比样本

```powershell
# 运行合并脚本
python merge_comparison_samples_only.py

# 预计时间：~5-10 分钟

# 输出：
📁 載入結果文件...
✓ Baseline 結果: 64,140 條記錄  # 10,690 × 6方法
  方法: ['identity', 'bicubic', 'gaussian', 'sharpen', 'unsharp', 'combined']
✓ Real-ESRGAN 結果: 1,500 條記錄  # 如果采样

🎲 選擇 50 張圖像生成對比樣本...
✓ 已選擇 50 張圖像
  Drone Motion Blur: 17 張
  Object Motion Blur: 17 張
  Low-light: 16 張

🖼️  生成對比樣本...
生成樣本: 100%|██████████| 50/50 [00:08<00:00, 5.23it/s]

✓ 對比樣本已保存至: D:\baseline_results_merged\comparison_samples

✅ 合併完成！

💡 說明:
  ✅ Leaderboards 保持獨立（未合併）
  ✅ 對比樣本包含 9 個方法
  ✅ Enhanced images 只在 Real-ESRGAN 資料夾
```

---

## 📁 最终文件结构

```
完整结构：

D:\baseline_results_old\
└── leaderboards\
    ├── full_results_with_gt.csv  # 传统方法完整结果
    ├── leaderboard_drone_motion_blur.csv
    ├── leaderboard_object_motion_blur.csv
    ├── leaderboard_low_light.csv
    ├── evaluation_summary.json
    └── comparison_samples\  # 8个方法（旧版）
        ├── xxx_comparison.png
        └── ...

D:\baseline_results_realesrgan_only\
├── checkpoint.json  # Checkpoint数据
├── sampled_images.txt  # 采样列表（1,500张）
├── enhanced_images\  # Real-ESRGAN增强图
│   ├── enhanced_xxx.png  # 1,500 张
│   └── ...
└── leaderboards\
    └── realesrgan_results.csv  # Real-ESRGAN结果

D:\baseline_results_merged\  # 新增
└── comparison_samples\  # 9个方法对比
    ├── xxx_comparison.png  # 50 张
    └── ...
```

---

## 🖼️ 对比样本内容

### 旧版（8个子图）

```
baseline_results_old/comparison_samples/xxx_comparison.png:

┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Original GT │ Degraded    │ identity    │ bicubic     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ gaussian    │ sharpen     │ unsharp     │ combined    │
└─────────────┴─────────────┴─────────────┴─────────────┘

8 个子图（2×4布局）
```

### 新版（9个子图）⭐

```
baseline_results_merged/comparison_samples/xxx_comparison.png:

┌─────────────┬─────────────┬─────────────┐
│ Original GT │ Degraded    │ identity    │
├─────────────┼─────────────┼─────────────┤
│ bicubic     │ gaussian    │ sharpen     │
├─────────────┼─────────────┼─────────────┤
│ unsharp     │ combined    │ realesrgan  │
└─────────────┴─────────────┴─────────────┘

9 个子图（3×3布局）
✅ 包含 Real-ESRGAN！
```

---

## 📊 结果查看方式

### 查看 Leaderboards

```powershell
# 传统方法排行榜
cat D:\baseline_results_old\leaderboards\leaderboard_*.csv

# Real-ESRGAN结果
cat D:\baseline_results_realesrgan_only\leaderboards\realesrgan_results.csv

# 对比分析（手动或写脚本）
```

### 查看对比样本

```powershell
# 旧版（8个方法）
explorer D:\baseline_results_old\leaderboards\comparison_samples\

# 新版（9个方法）⭐
explorer D:\baseline_results_merged\comparison_samples\

# 直接查看所有方法的视觉对比
```

### 查看增强图像

```powershell
# 只有 Real-ESRGAN 有
explorer D:\baseline_results_realesrgan_only\enhanced_images\

# 1,500 张（如果采样）或 10,690 张（如果全量）
```

---

## 🎯 核心配置

### merge_comparison_samples_only.py

```python
class MergeConfig:
    # 輸入路徑
    BASELINE_RESULTS_DIR = Path(r"D:\baseline_results_old\leaderboards")
    REALESRGAN_RESULTS_DIR = Path(r"D:\baseline_results_realesrgan_only")
    
    BASELINE_CSV = BASELINE_RESULTS_DIR / "full_results_with_gt.csv"
    REALESRGAN_CSV = REALESRGAN_RESULTS_DIR / "leaderboards" / "realesrgan_results.csv"
    REALESRGAN_ENHANCED_DIR = REALESRGAN_RESULTS_DIR / "enhanced_images"
    
    # 輸出路徑（只生成對比樣本）
    OUTPUT_DIR = Path(r"D:\baseline_results_merged")
    COMPARISON_DIR = OUTPUT_DIR / "comparison_samples"
    
    # 對比樣本配置
    NUM_SAMPLES = 50  # 生成 50 個對比樣本

可以修改的参数：
- NUM_SAMPLES：对比样本数量（默认50，可改为20、100等）
```

---

## ⚙️ 工作流程

### 完整流程

```bash
# 1. Baseline评估（已完成）
python baseline_evaluation_with_gt-HenryCC.py
# 输出：D:\baseline_results_old\

# 2. Real-ESRGAN评估（采样，推荐）
python evaluate_realesrgan_only_v2.py
# 配置：ENABLE_SAMPLING = True, SAMPLES_PER_TYPE = 500
# 时间：~24 小时
# 输出：D:\baseline_results_realesrgan_only\

# 3. 合并对比样本
python merge_comparison_samples_only.py
# 时间：~5-10 分钟
# 输出：D:\baseline_results_merged\comparison_samples\

# 完成！
```

---

## 🎓 为什么这个方案更好？

### 优点对比

```python
旧方案（merge_baseline_results.py）：
❌ 合并所有CSV → 结构混乱
❌ 重复Enhanced images → 浪费空间
❌ 不好区分各方法结果
❌ 修改原有结构

新方案（merge_comparison_samples_only.py）：✅
✅ 只合并对比样本 → 结构清晰
✅ Leaderboards独立 → 易于分析
✅ Enhanced images不重复 → 节省空间
✅ 不修改原有结构 → 安全
✅ 更符合实际使用需求
```

### 使用场景

```python
场景 1：查看各方法性能指标
→ 查看各自的 Leaderboards（CSV文件）
→ 独立分析，不混淆

场景 2：视觉对比所有方法
→ 查看 baseline_results_merged/comparison_samples/
→ 9个方法一目了然

场景 3：查看Real-ESRGAN增强图
→ 查看 baseline_results_realesrgan_only/enhanced_images/
→ 独立存储，不重复

场景 4：论文写作
→ Leaderboards：各自的CSV生成表格
→ Comparison samples：合并的对比图插入论文
→ 完美！✅
```

---

## 📝 论文中如何使用

### 表格（Leaderboards）

```latex
\begin{table}
\caption{Performance comparison of baseline methods}
\begin{tabular}{lcccc}
\hline
Method & PSNR (dB) & SSIM & LPIPS \\
\hline
Identity & 25.2 & 0.750 & 0.185 \\
Bicubic & 26.8 & 0.785 & 0.165 \\
... \\
\textbf{Real-ESRGAN} & \textbf{28.5} & \textbf{0.850} & \textbf{0.120} \\
\hline
\end{tabular}
\end{table}

数据来源：
- 传统方法：baseline_results_old/leaderboards/leaderboard_*.csv
- Real-ESRGAN：baseline_results_realesrgan_only/leaderboards/realesrgan_results.csv
```

### 对比图（Comparison Samples）

```latex
\begin{figure}
\includegraphics[width=\textwidth]{comparison_samples/xxx_comparison.png}
\caption{Visual comparison of different methods}
\end{figure}

图片来源：
baseline_results_merged/comparison_samples/
```

---

## 🆘 故障排除

### 问题 1：找不到原始图像

```python
错误：⚠️  找不到原始圖像

原因：
- 原始图像路径不对
- 文件名映射失败

解决：
检查配置：
ORIGINAL_DIR = Path(r"D:\FYP_Images")

确保文件存在
```

### 问题 2：Real-ESRGAN增强图不存在

```python
错误：⚠️  Real-ESRGAN 增強圖像不存在

原因：
- Real-ESRGAN评估时没有保存增强图
- 或者采样的图像与baseline不匹配

解决：
确保 evaluate_realesrgan_only_v2.py 中：
SAVE_ENHANCED_IMAGES = True
```

### 问题 3：CSV文件格式不对

```python
错误：KeyError: 'image_name' 或 'method'

原因：
- CSV格式与预期不符

解决：
检查CSV文件包含必要列：
- image_name
- method
- degradation_type
```

---

## 🎯 总结

### 你的方案（推荐）⭐⭐⭐

```python
特点：
✅ Leaderboards独立（各自分析）
✅ Comparison samples合并（统一对比）
✅ Enhanced images不重复（只在Real-ESRGAN文件夹）
✅ 结构清晰（不混乱）
✅ 论文友好（易于使用）

使用：
1. 运行 baseline（已完成）
2. 运行 Real-ESRGAN（~24小时）
3. 运行 merge_comparison_samples_only.py（~10分钟）

完成！
```

### 文件总览

```
脚本文件：
├── baseline_evaluation_with_gt-HenryCC.py  # 传统方法
├── evaluate_realesrgan_only_v2.py  # Real-ESRGAN
└── merge_comparison_samples_only.py  # 合并对比样本（新）✅

输出结构：
├── D:\baseline_results_old\leaderboards\  # 传统方法（独立）
├── D:\baseline_results_realesrgan_only\  # Real-ESRGAN（独立）
└── D:\baseline_results_merged\  # 合并对比样本（新）✅
```

---

## 🚀 立即使用

```powershell
# 步骤 1：运行 Real-ESRGAN（采样）
python evaluate_realesrgan_only_v2.py
# 时间：~24 小时

# 步骤 2：合并对比样本
python merge_comparison_samples_only.py
# 时间：~10 分钟

# 完成！查看结果：
explorer D:\baseline_results_merged\comparison_samples\
```

**完美方案！** 🎉

