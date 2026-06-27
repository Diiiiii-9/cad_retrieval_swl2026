# ==========================================
# Build FAISS databases for the pooling sweep checkpoints
# ==========================================
# Builds indexes for pooling_max and pooling_max_mean. The "mean" pooling
# database already exists at checkpoints\sweep\baseline_dim64_temp01
# (built during the dim/temp sweep) -- no need to rebuild it.
#
# IMPORTANT: --pooling must match what each config was TRAINED with, or
# retrieval.py will instantiate the wrong-shaped network and either crash
# or silently load garbage weights.
#
# Usage: from the repo root:
#   powershell -ExecutionPolicy Bypass -File build_pooling_databases.ps1

$configs = @(
    @{ name = "pooling_max";      out_dim = 64; pooling = "max" },
    @{ name = "pooling_max_mean"; out_dim = 64; pooling = "max_mean" }
)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

foreach ($cfg in $configs) {
    $name = $cfg.name
    $outDim = $cfg.out_dim
    $pooling = $cfg.pooling
    $saveDir = "checkpoints\sweep\$name"
    $logFile = "logs\build_$name.log"
    $indexFile = "$saveDir\faiss_index.bin"

    if (Test-Path $indexFile) {
        Write-Host ("[SKIP] " + $name + " already has a faiss_index.bin")
        continue
    }

    Write-Host ("[BUILD] " + $name + "  (out_dim=" + $outDim + ", pooling=" + $pooling + ")")

    $cmd = "python -m src.retrieval --build --save_dir $saveDir --out_dim $outDim --pooling $pooling"
    Invoke-Expression "$cmd *>&1 | Tee-Object -FilePath `"$logFile`""

    if (Test-Path $indexFile) {
        Write-Host ("[DONE] " + $name + " -> " + $indexFile)
    } else {
        Write-Host ("[WARNING] " + $name + " did not produce a faiss_index.bin -- check " + $logFile)
    }
}

Write-Host ""
Write-Host "All builds attempted. Check above for any [WARNING] lines before evaluating."
