# Terminal Commands for Degradation Synthesis

## ✅ Installation Commands (Already Completed)

```bash
# Navigate to project directory
cd "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP"

# 1. Create virtual environment
python3 -m venv venv_degradation

# 2. Activate virtual environment
source venv_degradation/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (CPU version for Mac)
pip install torch torchvision torchaudio

# 5. Install other dependencies
pip install -r requirements_degradation.txt

# 6. Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

## ✅ Test Run (Already Completed - Successful!)

```bash
# Test with 3 images from each split
source venv_degradation/bin/activate
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./test_degraded_output \
    --max_images 3 \
    --device cpu
```

**Result**: ✅ Generated 19 degraded images successfully!
- Global blur: 9 images
- Object blur: 1 image (only when objects detected)
- Low-light: 9 images

## 📊 Check Test Results

```bash
# View summary
cat test_degraded_output/degradation_summary.json

# List degraded images
ls -lh test_degraded_output/degraded/

# Count files
ls test_degraded_output/degraded/ | wc -l

# View pairs CSV
head -5 test_degraded_output/pairs.csv
```

## 🚀 Next Steps - Process More Images

### Option 1: Process 10 images (Quick Test)
```bash
source venv_degradation/bin/activate
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_10 \
    --max_images 10 \
    --device cpu
```

### Option 2: Process 100 images (Medium Test)
```bash
source venv_degradation/bin/activate
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_100 \
    --max_images 100 \
    --device cpu
```

### Option 3: Process ALL images (Full Dataset) - Standard Version
```bash
# This will take 8-12 hours on MacBook
source venv_degradation/bin/activate
python degradation_synthesis.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --device cpu
```

### Option 4: Process ALL images - Parallel Version (RECOMMENDED for Mac!)
```bash
# 2-4x faster! Uses all CPU cores
source venv_degradation/bin/activate
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data
```

### Option 5: FASTEST for Mac - Skip Object Blur + Parallel
```bash
# 6-8x faster! Only global blur + low-light
source venv_degradation/bin/activate
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --no_object_blur
```

## 🎯 Test Single Image with Visualization

```bash
# Pick any image from your dataset
source venv_degradation/bin/activate
python test_degradation.py \
    --image "/Users/henrysze/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/Y4_SEM1/FYP_Images/Images_Aerial-Traffic/img_1.jpg" \
    --output ./single_image_test \
    --device cpu

# View the comparison grid
open ./single_image_test/comparison_grid.jpg
```

## 📋 Useful Monitoring Commands

### Monitor Progress (While Running)
```bash
# In another terminal window
# Check CPU usage
top

# Or use Activity Monitor
open -a "Activity Monitor"

# Watch output directory grow
watch -n 5 'ls degraded_data/degraded/ | wc -l'
```

### Run in Background (For Long Processing)
```bash
# Run in background, save output to log
source venv_degradation/bin/activate
nohup python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --no_object_blur > degradation.log 2>&1 &

# Check progress
tail -f degradation.log

# Check if still running
ps aux | grep degradation
```

## 🧹 Cleanup Test Files (Optional)

```bash
# Remove test output if you want to start fresh
rm -rf test_degraded_output
rm -rf degraded_10
rm -rf single_image_test
```

## ⚙️ Virtual Environment Management

```bash
# Activate venv (do this every time you open a new terminal)
source venv_degradation/bin/activate

# Deactivate venv (when done)
deactivate

# Remove venv (if you want to reinstall)
rm -rf venv_degradation
```

## 📈 Expected Processing Times on Your MacBook

Based on your setup (Apple Silicon Mac):

| Images | Standard Version | Parallel Version | Parallel (No Object Blur) |
|--------|-----------------|------------------|---------------------------|
| 3      | ~3 seconds      | ~2 seconds       | ~1 second                 |
| 10     | ~10 seconds     | ~5 seconds       | ~3 seconds                |
| 100    | ~2 minutes      | ~1 minute        | ~30 seconds               |
| 1000   | ~20 minutes     | ~10 minutes      | ~5 minutes                |
| 4391 (All) | 8-12 hours  | 4-6 hours        | 2-3 hours                 |

## 🎓 Summary of What Works

✅ Virtual environment created  
✅ PyTorch installed (CPU version, MPS available)  
✅ YOLOv8 installed  
✅ Test run successful (3 images → 19 degraded images)  
✅ All 3 degradation types working:
  - Global motion blur ✓
  - Object motion blur ✓ (detected 1 object in test)
  - Low-light degradation ✓

## 🚀 Recommended Command for You

Based on your Mac setup, I recommend this command for processing your full dataset:

```bash
source venv_degradation/bin/activate
python degradation_synthesis_parallel.py \
    --split_dir ./data_split_results \
    --output_dir ./degraded_data \
    --no_object_blur
```

This will:
- Use all your CPU cores (parallel processing)
- Skip the slowest part (object blur)
- Complete in ~2-3 hours instead of 8-12 hours
- Still generate ~8,000-9,000 high-quality degraded images

---

**Note**: You're currently on macOS, so all commands use forward slashes `/` and `source` to activate the venv. If you switch to Windows, the commands will be slightly different (use backslashes `\` and `venv_degradation\Scripts\activate`).



