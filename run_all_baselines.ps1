# 一鍵運行所有基準測試
# Run All Baselines - PowerShell Script
# =====================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  基準測試自動化腳本" -ForegroundColor Cyan
Write-Host "  Baseline Testing Automation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 設置錯誤處理
$ErrorActionPreference = "Continue"

# 項目目錄
$FYP_DIR = "C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP"
$BASELINE_ENV = "$FYP_DIR\venv_baseline"
$REALESRGAN_ENV = "$FYP_DIR\venv_realesrgan"

# 檢查是否在正確的目錄
if ((Get-Location).Path -ne $FYP_DIR) {
    Write-Host "切換到 FYP 目錄..." -ForegroundColor Yellow
    Set-Location $FYP_DIR
}

# ========================================
# 函數定義
# ========================================

function Test-VirtualEnv {
    param($EnvPath)
    return Test-Path "$EnvPath\Scripts\Activate.ps1"
}

function Show-Menu {
    Write-Host "`n請選擇要運行的測試：" -ForegroundColor Green
    Write-Host "  1. 僅運行簡單基準測試（推薦，快速）" -ForegroundColor White
    Write-Host "  2. 僅運行 Real-ESRGAN 測試（需要先設置）" -ForegroundColor White
    Write-Host "  3. 運行全部測試（簡單 + Real-ESRGAN）" -ForegroundColor White
    Write-Host "  4. 測試模式（僅處理少量圖像）" -ForegroundColor White
    Write-Host "  5. 查看結果" -ForegroundColor White
    Write-Host "  6. 設置環境（首次運行）" -ForegroundColor White
    Write-Host "  0. 退出" -ForegroundColor White
    Write-Host ""
}

function Setup-BaselineEnv {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "設置基準測試環境" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (-not (Test-VirtualEnv $BASELINE_ENV)) {
        Write-Host "`n創建虛擬環境: venv_baseline..." -ForegroundColor Yellow
        python -m venv venv_baseline
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 虛擬環境創建成功" -ForegroundColor Green
        } else {
            Write-Host "✗ 虛擬環境創建失敗" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "✓ 虛擬環境已存在" -ForegroundColor Green
    }
    
    Write-Host "`n啟動虛擬環境並安裝依賴..." -ForegroundColor Yellow
    
    # 啟動環境並安裝
    & "$BASELINE_ENV\Scripts\Activate.ps1"
    
    Write-Host "安裝 PyTorch..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    Write-Host "安裝其他依賴..." -ForegroundColor Yellow
    pip install -r requirements_baseline.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ 基準測試環境設置完成！" -ForegroundColor Green
        return $true
    } else {
        Write-Host "`n✗ 依賴安裝失敗" -ForegroundColor Red
        return $false
    }
}

function Run-SimpleBaselines {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "運行簡單基準測試" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (-not (Test-VirtualEnv $BASELINE_ENV)) {
        Write-Host "✗ 虛擬環境不存在，請先運行選項 6 設置環境" -ForegroundColor Red
        return
    }
    
    Write-Host "啟動環境: venv_baseline" -ForegroundColor Yellow
    & "$BASELINE_ENV\Scripts\Activate.ps1"
    
    Write-Host "開始處理..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    python baseline_evaluation.py
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ 簡單基準測試完成！" -ForegroundColor Green
        Write-Host "耗時: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        Write-Host "結果目錄: D:\baseline_results\" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ 測試過程中出現錯誤" -ForegroundColor Red
    }
    
    deactivate
}

function Run-RealESRGAN {
    param([bool]$TestMode = $false)
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "運行 Real-ESRGAN 測試" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (-not (Test-VirtualEnv $REALESRGAN_ENV)) {
        Write-Host "✗ Real-ESRGAN 環境不存在" -ForegroundColor Red
        Write-Host "請參考 setup_realesrgan_env.md 手動設置" -ForegroundColor Yellow
        return
    }
    
    # 檢查模型文件
    $modelPath = "D:\Real-ESRGAN\experiments\pretrained_models\RealESRGAN_x4plus.pth"
    if (-not (Test-Path $modelPath)) {
        Write-Host "✗ 模型文件不存在: $modelPath" -ForegroundColor Red
        Write-Host "請下載模型並放置到指定位置" -ForegroundColor Yellow
        Write-Host "下載連結: https://github.com/xinntao/Real-ESRGAN/releases" -ForegroundColor Cyan
        return
    }
    
    Write-Host "啟動環境: venv_realesrgan" -ForegroundColor Yellow
    & "$REALESRGAN_ENV\Scripts\Activate.ps1"
    
    Write-Host "開始處理..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    if ($TestMode) {
        Write-Host "⚠️  測試模式：僅處理 5 張圖像" -ForegroundColor Yellow
        python realesrgan_baseline.py --test
    } else {
        python realesrgan_baseline.py
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Real-ESRGAN 測試完成！" -ForegroundColor Green
        Write-Host "耗時: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        Write-Host "結果目錄: D:\realesrgan_results\" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ 測試過程中出現錯誤" -ForegroundColor Red
    }
    
    deactivate
}

function Show-Results {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "查看結果" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    Write-Host "`n選擇要查看的結果：" -ForegroundColor Green
    Write-Host "  1. 簡單基準測試結果" -ForegroundColor White
    Write-Host "  2. Real-ESRGAN 結果" -ForegroundColor White
    Write-Host "  3. 返回主菜單" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "請輸入選項"
    
    switch ($choice) {
        "1" {
            if (Test-Path "D:\baseline_results") {
                Write-Host "`n打開結果目錄..." -ForegroundColor Yellow
                explorer "D:\baseline_results\leaderboards"
                Start-Sleep -Seconds 1
                explorer "D:\baseline_results\comparison_samples"
            } else {
                Write-Host "✗ 結果目錄不存在，請先運行基準測試" -ForegroundColor Red
            }
        }
        "2" {
            if (Test-Path "D:\realesrgan_results") {
                Write-Host "`n打開結果目錄..." -ForegroundColor Yellow
                explorer "D:\realesrgan_results\leaderboards"
                Start-Sleep -Seconds 1
                explorer "D:\realesrgan_results\comparisons"
            } else {
                Write-Host "✗ 結果目錄不存在，請先運行 Real-ESRGAN 測試" -ForegroundColor Red
            }
        }
    }
}

function Run-TestMode {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "測試模式（快速驗證）" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    Write-Host "`n選擇要測試的項目：" -ForegroundColor Green
    Write-Host "  1. 測試簡單基準（處理所有圖像，但限制樣本生成）" -ForegroundColor White
    Write-Host "  2. 測試 Real-ESRGAN（僅處理 5 張圖像）" -ForegroundColor White
    Write-Host "  3. 返回主菜單" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "請輸入選項"
    
    switch ($choice) {
        "1" { Run-SimpleBaselines }
        "2" { Run-RealESRGAN -TestMode $true }
    }
}

# ========================================
# 主程序
# ========================================

# 檢查 Python
Write-Host "檢查 Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python 可用: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python 未安裝或不在 PATH 中" -ForegroundColor Red
    Write-Host "請先安裝 Python 3.7 或更高版本" -ForegroundColor Yellow
    exit 1
}

# 檢查退化圖像目錄
Write-Host "檢查退化圖像目錄..." -ForegroundColor Yellow
if (Test-Path "D:\degraded_full_dataset") {
    $imageCount = (Get-ChildItem -Path "D:\degraded_full_dataset" -Recurse -File -Include *.jpg,*.png).Count
    Write-Host "✓ 退化圖像目錄存在，共 $imageCount 張圖像" -ForegroundColor Green
} else {
    Write-Host "✗ 退化圖像目錄不存在: D:\degraded_full_dataset" -ForegroundColor Red
    Write-Host "請先運行退化合成: python degradation_synthesis_parallel.py" -ForegroundColor Yellow
}

# 主循環
while ($true) {
    Show-Menu
    $choice = Read-Host "請輸入選項"
    
    switch ($choice) {
        "1" { 
            Run-SimpleBaselines
            Write-Host "`n按任意鍵繼續..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
        "2" { 
            Run-RealESRGAN
            Write-Host "`n按任意鍵繼續..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
        "3" { 
            Run-SimpleBaselines
            Run-RealESRGAN
            Write-Host "`n按任意鍵繼續..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
        "4" { 
            Run-TestMode
            Write-Host "`n按任意鍵繼續..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
        "5" { 
            Show-Results
            Write-Host "`n按任意鍵繼續..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
        "6" { 
            $success = Setup-BaselineEnv
            if ($success) {
                Write-Host "`n✓ 環境設置完成！現在可以運行選項 1 進行測試" -ForegroundColor Green
            }
            Write-Host "`n按任意鍵繼續..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
        "0" { 
            Write-Host "`n再見！" -ForegroundColor Cyan
            exit 0
        }
        default { 
            Write-Host "`n✗ 無效的選項，請重新選擇" -ForegroundColor Red
        }
    }
}

