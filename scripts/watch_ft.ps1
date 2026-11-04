# Live view of the fp-v4.1 pipeline: spatial fine-tune -> store rebuild -> full-model head. Read-only.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_ft.ps1
param([int]$EverySeconds = 20)

$logs = Join-Path (Split-Path $PSScriptRoot -Parent) "results\training\logs"
$stages = [ordered]@{
    "1 Fine-tune spatial EfficientNet (batch 64)" = "train_ft_finetune_spatial.log"
    "2 Re-extract spatial features + new store"  = "train_ft2_build_ft_store.log"
    "3 Train full 5-pillar head (fp-v4.1)"       = "train_ft2_group_p1_store_results_training_feature_store_ft_runs_results_training_runs_ft.log"
}
while ($true) {
    Clear-Host
    Write-Host "Pixel-Artifacts fp-v4.1 pipeline   $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    foreach ($s in $stages.Keys) {
        $f = Join-Path $logs $stages[$s]
        Write-Host "`n$s" -ForegroundColor Yellow
        if (-not (Test-Path $f)) { Write-Host "  waiting" -ForegroundColor DarkGray; continue }
        $lines = Get-Content $f | ? { $_ -match "epoch|rows|written|TEST|done|Error|Traceback|resumed|Paused" }
        $lines | Select-Object -Last 6 | % {
            $c = if ($_ -match "Error|Traceback") { "Red" } elseif ($_ -match "\*best\*|TEST|done") { "Green" } else { "Gray" }
            Write-Host "  $_" -ForegroundColor $c
        }
    }
    Write-Host "`nQueues:" -ForegroundColor Yellow
    foreach ($q in "queue_ft.log", "queue_ft2.log") {
        $p = Join-Path $logs $q
        if (Test-Path $p) { Get-Content $p -Tail 1 | % { Write-Host "  $q  $_" } }
    }
    Start-Sleep $EverySeconds
}
