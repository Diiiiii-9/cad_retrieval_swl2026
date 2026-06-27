# ==========================================
# Augmentation Mode Ablation Sweep
# ==========================================
# Trains 2 new configs (edge_drop, feature_noise). The "both" mode is NOT
# retrained here -- it's already covered by checkpoints\sweep\baseline_dim64_temp01
# (out_dim=64, temp=0.1, aug_mode defaulted to "both", 30 epochs). Reusing it
# keeps the comparison valid since every other setting is identical -- aug_mode
# is the only thing that differs across all 3 configs in this ablation.
#
# Usage: from the repo root:
#   powershell -ExecutionPolicy Bypass -File run_aug_sweep.ps1

$EPOCHS = 30   # MUST match the other sweeps' epoch count for valid comparison

$configs = @(
    @{ name = "aug_edge_drop";     args = "--out_dim 64 --temp 0.1 --aug_mode edge_drop" },
    @{ name = "aug_feature_noise"; args = "--out_dim 64 --temp 0.1 --aug_mode feature_noise" }
)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints\sweep" | Out-Null

$totalStart = Get-Date
$totalStartStr = $totalStart.ToString("yyyy-MM-dd HH:mm:ss")

Write-Host "=============================================="
Write-Host ("Starting augmentation sweep: " + $configs.Count + " new configs at " + $EPOCHS + " epochs each")
Write-Host ("(aug_mode=both reused from checkpoints\sweep\baseline_dim64_temp01)")
Write-Host ("Start time: " + $totalStartStr)
Write-Host "=============================================="

foreach ($cfg in $configs) {
    $name = $cfg.name
    $saveDir = "checkpoints\sweep\$name"
    $logFile = "logs\$name.log"
    $doneMarker = "$saveDir\final_encoder_weights.pth"

    if (Test-Path $doneMarker) {
        Write-Host ("[SKIP] " + $name + " already completed (found " + $doneMarker + ")")
        continue
    }

    Write-Host ""
    Write-Host "----------------------------------------------"
    Write-Host ("[RUN] " + $name + "  (" + $cfg.args + " --epochs " + $EPOCHS + ")")
    $startedStr = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host ("Started: " + $startedStr)
    Write-Host ("Log: " + $logFile)
    Write-Host "----------------------------------------------"

    $runStart = Get-Date

    $cmd = "python -m src.train_unsupervised --epochs $EPOCHS $($cfg.args) --save_dir $saveDir"
    Invoke-Expression "$cmd *>&1 | Tee-Object -FilePath `"$logFile`""

    $runEnd = Get-Date
    $elapsedSeconds = ($runEnd - $runStart).TotalSeconds
    $elapsedStr = [string]::Format("{0:00}:{1:00}:{2:00}", [math]::Floor($elapsedSeconds/3600), [math]::Floor(($elapsedSeconds%3600)/60), [math]::Floor($elapsedSeconds%60))

    if (Test-Path $doneMarker) {
        Write-Host ("[DONE] " + $name + " completed in " + $elapsedStr)
    } else {
        Write-Host ("[WARNING] " + $name + " finished running but no final_encoder_weights.pth found -- check " + $logFile + " for errors")
    }
}

$totalEnd = Get-Date
$totalElapsedSeconds = ($totalEnd - $totalStart).TotalSeconds
$totalElapsedStr = [string]::Format("{0:00}:{1:00}:{2:00}", [math]::Floor($totalElapsedSeconds/3600), [math]::Floor(($totalElapsedSeconds%3600)/60), [math]::Floor($totalElapsedSeconds%60))
$totalEndStr = $totalEnd.ToString("yyyy-MM-dd HH:mm:ss")

Write-Host ""
Write-Host "=============================================="
Write-Host ("Augmentation sweep finished (or stopped) at " + $totalEndStr)
Write-Host ("Total elapsed: " + $totalElapsedStr)
Write-Host "=============================================="
