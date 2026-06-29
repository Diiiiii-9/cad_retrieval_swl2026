# ==========================================
# Dimension + Temperature Ablation Sweep
# ==========================================
# Runs 6 training configs sequentially, all at a FIXED, reduced epoch count
# (single-variable rule: epochs is constant across the whole sweep, never
# varied per-config). Each run logs to its own file under logs/ so you can
# check progress or diagnose a crash without losing prior results.
#
# Usage (works from any directory; the script switches to the repo root itself), run:
#   powershell -ExecutionPolicy Bypass -File scripts\run_sweep_fixed.ps1
#
# If it's interrupted, just re-run it -- already-completed configs are
# skipped automatically (checked via existence of final_encoder_weights.pth).

$EPOCHS = 30   # FIXED for the entire sweep -- do not change between configs

# Each entry: (name, extra_args)
# Anchor to the repo root regardless of where this script is launched from
# (this file lives in scripts/, one level below the repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$configs = @(
    @{ name = "baseline_dim64_temp01"; args = "--out_dim 64  --temp 0.1" },
    @{ name = "dim32";                 args = "--out_dim 32  --temp 0.1" },
    @{ name = "dim128";                args = "--out_dim 128 --temp 0.1" },
    @{ name = "dim256";                args = "--out_dim 256 --temp 0.1" },
    @{ name = "temp005";               args = "--out_dim 64  --temp 0.05" },
    @{ name = "temp05";                args = "--out_dim 64  --temp 0.5" }
)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints\sweep" | Out-Null

$totalStart = Get-Date
$totalStartStr = $totalStart.ToString("yyyy-MM-dd HH:mm:ss")

Write-Host "=============================================="
Write-Host ("Starting sweep: " + $configs.Count + " configs at " + $EPOCHS + " epochs each")
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
Write-Host ("Sweep finished (or stopped) at " + $totalEndStr)
Write-Host ("Total elapsed: " + $totalElapsedStr)
Write-Host "=============================================="
