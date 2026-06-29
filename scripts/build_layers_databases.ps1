# ==========================================
# Build FAISS databases for the layer-count sweep checkpoints
# ==========================================
# Builds indexes for layers2 and layers5. The num_layers=3 database already
# exists at checkpoints\sweep\baseline_dim64_temp01 -- no need to rebuild it.
#
# IMPORTANT: --num_layers must match what each config was TRAINED with, or
# retrieval.py will instantiate the wrong-shaped network (wrong number of
# GINE layers) and fail to load the weights correctly.
#
# Usage (works from any directory; the script switches to the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts\build_layers_databases.ps1

# Anchor to the repo root regardless of where this script is launched from
# (this file lives in scripts/, one level below the repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$configs = @(
    @{ name = "layers2"; out_dim = 64; num_layers = 2 },
    @{ name = "layers5"; out_dim = 64; num_layers = 5 }
)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

foreach ($cfg in $configs) {
    $name = $cfg.name
    $outDim = $cfg.out_dim
    $numLayers = $cfg.num_layers
    $saveDir = "checkpoints\sweep\$name"
    $logFile = "logs\build_$name.log"
    $indexFile = "$saveDir\faiss_index.bin"

    if (Test-Path $indexFile) {
        Write-Host ("[SKIP] " + $name + " already has a faiss_index.bin")
        continue
    }

    Write-Host ("[BUILD] " + $name + "  (out_dim=" + $outDim + ", num_layers=" + $numLayers + ")")

    $cmd = "python -m src.retrieval --build --save_dir $saveDir --out_dim $outDim --num_layers $numLayers"
    Invoke-Expression "$cmd *>&1 | Tee-Object -FilePath `"$logFile`""

    if (Test-Path $indexFile) {
        Write-Host ("[DONE] " + $name + " -> " + $indexFile)
    } else {
        Write-Host ("[WARNING] " + $name + " did not produce a faiss_index.bin -- check " + $logFile)
    }
}

Write-Host ""
Write-Host "All builds attempted. Check above for any [WARNING] lines before evaluating."
