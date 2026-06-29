# ==========================================
# Build FAISS databases for all 6 sweep checkpoints
# ==========================================
# For each trained config, runs retrieval.py --build against its own
# checkpoint folder, using the SAME out_dim that config was trained with.
# This produces faiss_index.bin + metadata.pkl inside each
# checkpoints\sweep\<name>\ folder, ready for evaluation.
#
# Usage (works from any directory; the script switches to the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts\build_sweep_databases.ps1

# Anchor to the repo root regardless of where this script is launched from
# (this file lives in scripts/, one level below the repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$configs = @(
    @{ name = "baseline_dim64_temp01"; out_dim = 64 },
    @{ name = "dim32";                 out_dim = 32 },
    @{ name = "dim128";                out_dim = 128 },
    @{ name = "dim256";                out_dim = 256 },
    @{ name = "temp005";               out_dim = 64 },
    @{ name = "temp05";                out_dim = 64 }
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
