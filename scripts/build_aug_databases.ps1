# ==========================================
# Build FAISS databases for the augmentation sweep checkpoints
# ==========================================
# Builds indexes for aug_edge_drop and aug_feature_noise. The "both" mode
# database already exists at checkpoints\sweep\baseline_dim64_temp01 -- no
# need to rebuild it.
#
# NOTE: aug_mode does NOT affect the network architecture (only how training
# data is perturbed), so retrieval.py needs no --aug_mode flag -- the same
# out_dim/pooling defaults as the baseline apply here.
#
# Usage (works from any directory; the script switches to the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts\build_aug_databases.ps1

# Anchor to the repo root regardless of where this script is launched from
# (this file lives in scripts/, one level below the repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$configs = @(
    @{ name = "aug_edge_drop";     out_dim = 64 },
    @{ name = "aug_feature_noise"; out_dim = 64 }
)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

foreach ($cfg in $configs) {
    $name = $cfg.name
    $outDim = $cfg.out_dim
    $saveDir = "checkpoints\sweep\$name"
    $logFile = "logs\build_$name.log"
    $indexFile = "$saveDir\faiss_index.bin"

    if (Test-Path $indexFile) {
        Write-Host ("[SKIP] " + $name + " already has a faiss_index.bin")
        continue
    }

    Write-Host ("[BUILD] " + $name + "  (out_dim=" + $outDim + ")")

    $cmd = "python -m src.retrieval --build --save_dir $saveDir --out_dim $outDim"
    Invoke-Expression "$cmd *>&1 | Tee-Object -FilePath `"$logFile`""

    if (Test-Path $indexFile) {
        Write-Host ("[DONE] " + $name + " -> " + $indexFile)
    } else {
        Write-Host ("[WARNING] " + $name + " did not produce a faiss_index.bin -- check " + $logFile)
    }
}

Write-Host ""
Write-Host "All builds attempted. Check above for any [WARNING] lines before evaluating."
