# ==========================================
# Layer Count Ablation Sweep
# ==========================================
# Trains 2 new configs (num_layers=2, num_layers=5). The 3-layer config is
# NOT retrained here -- it's already covered by checkpoints\sweep\baseline_dim64_temp01
# (out_dim=64, temp=0.1, num_layers defaulted to 3, 30 epochs). Reusing it
# keeps the comparison valid since every other setting is identical --
# num_layers is the only thing that differs across all 3 configs here.
#
# Usage (works from any directory; the script switches to the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts\run_layers_sweep.ps1

$EPOCHS = 30   # MUST match the other sweeps' epoch count for valid comparison

# Anchor to the repo root regardless of where this script is launched from
# (this file lives in scripts/, one level below the repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$configs = @(
    @{ name = "layers2"; args = "--out_dim 64 --temp 0.1 --num_layers 2" },
    @{ name = "layers5"; args = "--out_dim 64 --temp 0.1 --num_layers 5" }
)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints\sweep" | Out-Null

$totalStart = Get-Date
$totalStartStr = $totalStart.ToString("yyyy-MM-dd HH:mm:ss")

Write-Host "=============================================="
Write-Host ("Starting layer-count sweep: " + $configs.Count + " new configs at " + $EPOCHS + " epochs each")
Write-Host ("(num_layers=3 reused from checkpoints\sweep\baseline_dim64_temp01)")
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
Write-Host ("Layer-count sweep finished (or stopped) at " + $totalEndStr)
Write-Host ("Total elapsed: " + $totalElapsedStr)
Write-Host "=============================================="
