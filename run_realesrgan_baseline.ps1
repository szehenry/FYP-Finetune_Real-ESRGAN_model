# Real-ESRGAN 基準評估運行腳本
# =====================================
# 自動化運行 Real-ESRGAN 基準評估流程
#
# 使用方式：
#   .\run_realesrgan_baseline.ps1
#
# 或選擇特定步驟：
#   .\run_realesrgan_baseline.ps1 -Step test     # 只測試
#   .\run_realesrgan_baseline.ps1 -Step eval     # 只評估
#   .\run_realesrgan_baseline.ps1 -Step merge    # 只合併
#
# 作者：FYP Project
# 日期：2025-12-22

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('all', 'test', 'eval', 'merge')]
    [string]$Step = 'all'
)

# 設置錯誤處理
$ErrorActionPreference = "Stop"

# 顏色輸出函數
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# 檢查虛擬環境
function Check-VirtualEnv {
    if (-not (Test-Path ".\venv_baseline\Scripts\Activate.ps1")) {
        Write-ColorOutput "❌ 找不到虛擬環境: venv_baseline" "Red"
        Write-ColorOutput "請先創建虛擬環境或確認路徑正確" "Yellow"
        exit 1
    }
}

# 激活虛擬環境
function Activate-VirtualEnv {
    Write-ColorOutput "`n🔧 激活虛擬環境..." "Cyan"
    & ".\venv_baseline\Scripts\Activate.ps1"
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "❌ 虛擬環境激活失敗" "Red"
        exit 1
    }
    
    Write-ColorOutput "✓ 虛擬環境已激活" "Green"
}

# 檢查 Python 腳本
function Check-Scripts {
    $scripts = @(
        "test_realesrgan_baseline.py",
        "evaluate_realesrgan_only.py",
        "merge_baseline_results.py"
    )
    
    foreach ($script in $scripts) {
        if (-not (Test-Path $script)) {
            Write-ColorOutput "❌ 找不到腳本: $script" "Red"
            exit 1
        }
    }
}

# 檢查必要路徑
function Check-Paths {
    Write-ColorOutput "`n🔍 檢查必要路徑..." "Cyan"
    
    $paths = @{
        "配對文件" = "D:\degraded_full_dataset\pairs.csv"
        "退化圖像" = "D:\degraded_full_dataset\degraded"
        "原始圖像" = "D:\FYP_Images"
        "Real-ESRGAN 模型" = "D:\Real-ESRGAN\experiments\pretrained_models\RealESRGAN_x4plus.pth"
    }
    
    $allGood = $true
    foreach ($item in $paths.GetEnumerator()) {
        if (Test-Path $item.Value) {
            Write-ColorOutput "  ✓ $($item.Key): 存在" "Green"
        } else {
            Write-ColorOutput "  ❌ $($item.Key): 不存在" "Red"
            Write-ColorOutput "     路徑: $($item.Value)" "Yellow"
            $allGood = $false
        }
    }
    
    if (-not $allGood) {
        Write-ColorOutput "`n⚠️  某些必要路徑不存在，請檢查配置" "Yellow"
        $continue = Read-Host "是否繼續？ (y/n)"
        if ($continue -ne 'y') {
            exit 1
        }
    }
}

# 運行測試
function Run-Test {
    Write-ColorOutput "`n" "White"
    Write-ColorOutput "=" * 80 "Cyan"
    Write-ColorOutput "🧪 步驟 1/3: 測試流程（10 張圖像）" "Cyan"
    Write-ColorOutput "=" * 80 "Cyan"
    Write-ColorOutput "預計時間: 5-10 分鐘`n" "Yellow"
    
    python test_realesrgan_baseline.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "`n❌ 測試失敗" "Red"
        exit 1
    }
    
    Write-ColorOutput "`n✓ 測試完成" "Green"
}

# 運行評估
function Run-Evaluation {
    Write-ColorOutput "`n" "White"
    Write-ColorOutput "=" * 80 "Cyan"
    Write-ColorOutput "🎯 步驟 2/3: 評估 Real-ESRGAN" "Cyan"
    Write-ColorOutput "=" * 80 "Cyan"
    Write-ColorOutput "預計時間: 2-3 小時`n" "Yellow"
    
    $startTime = Get-Date
    Write-ColorOutput "開始時間: $($startTime.ToString('HH:mm:ss'))" "Gray"
    
    python evaluate_realesrgan_only.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "`n❌ 評估失敗" "Red"
        exit 1
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-ColorOutput "`n✓ 評估完成" "Green"
    Write-ColorOutput "耗時: $($duration.Hours) 小時 $($duration.Minutes) 分鐘" "Gray"
}

# 運行合併
function Run-Merge {
    Write-ColorOutput "`n" "White"
    Write-ColorOutput "=" * 80 "Cyan"
    Write-ColorOutput "🔗 步驟 3/3: 合併結果並生成對比圖" "Cyan"
    Write-ColorOutput "=" * 80 "Cyan"
    Write-ColorOutput "預計時間: 10-15 分鐘`n" "Yellow"
    
    $startTime = Get-Date
    
    python merge_baseline_results.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "`n❌ 合併失敗" "Red"
        exit 1
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-ColorOutput "`n✓ 合併完成" "Green"
    Write-ColorOutput "耗時: $($duration.Minutes) 分鐘" "Gray"
}

# 顯示結果
function Show-Results {
    Write-ColorOutput "`n" "White"
    Write-ColorOutput "=" * 80 "Green"
    Write-ColorOutput "✅ 所有任務完成！" "Green"
    Write-ColorOutput "=" * 80 "Green"
    
    Write-ColorOutput "`n📊 結果位置:" "Cyan"
    Write-ColorOutput "  完整結果: D:\baseline_results_merged\leaderboards\full_results_merged.csv" "White"
    Write-ColorOutput "  對比圖:   D:\baseline_results_merged\comparison_samples\" "White"
    Write-ColorOutput "  排行榜:   D:\baseline_results_merged\leaderboards\leaderboard_*.csv" "White"
    
    Write-ColorOutput "`n💡 下一步:" "Cyan"
    Write-ColorOutput "  1. 查看排行榜，分析 Real-ESRGAN 性能" "White"
    Write-ColorOutput "  2. 檢查對比圖（50 張，包含 9 個方法）" "White"
    Write-ColorOutput "  3. 開始微調 Real-ESRGAN（如果需要）" "White"
    Write-ColorOutput "  4. 撰寫 FYP 論文" "White"
}

# 主函數
function Main {
    Write-ColorOutput "`n" "White"
    Write-ColorOutput "=" * 80 "Magenta"
    Write-ColorOutput "🚀 Real-ESRGAN 基準評估自動化腳本" "Magenta"
    Write-ColorOutput "=" * 80 "Magenta"
    
    # 檢查環境
    Check-VirtualEnv
    Check-Scripts
    Check-Paths
    
    # 激活虛擬環境
    Activate-VirtualEnv
    
    # 記錄總開始時間
    $totalStartTime = Get-Date
    
    # 根據參數執行步驟
    switch ($Step) {
        'test' {
            Run-Test
        }
        'eval' {
            Run-Evaluation
        }
        'merge' {
            Run-Merge
        }
        'all' {
            # 詢問是否跳過測試
            Write-ColorOutput "`n❓ 是否運行測試（推薦）？" "Yellow"
            Write-ColorOutput "   測試只需 5-10 分鐘，可以確保流程正常" "Gray"
            $runTest = Read-Host "運行測試？ (y/n，默認 y)"
            
            if ($runTest -ne 'n') {
                Run-Test
                
                Write-ColorOutput "`n⏸️  測試完成，準備開始完整評估..." "Yellow"
                Write-ColorOutput "   這將需要 2-3 小時，確保您有足夠時間" "Gray"
                $continue = Read-Host "繼續？ (y/n)"
                
                if ($continue -ne 'y') {
                    Write-ColorOutput "`n⏹️  已取消" "Yellow"
                    exit 0
                }
            }
            
            Run-Evaluation
            Run-Merge
        }
    }
    
    # 計算總耗時
    $totalEndTime = Get-Date
    $totalDuration = $totalEndTime - $totalStartTime
    
    # 顯示結果
    Show-Results
    
    Write-ColorOutput "`n⏱️  總耗時: $($totalDuration.Hours) 小時 $($totalDuration.Minutes) 分鐘" "Gray"
}

# 運行主函數
try {
    Main
} catch {
    Write-ColorOutput "`n❌ 發生錯誤: $_" "Red"
    Write-ColorOutput $_.ScriptStackTrace "Red"
    exit 1
}

