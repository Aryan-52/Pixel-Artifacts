# Live progress for head training (reads the training log; does not touch the job).
#   powershell -ExecutionPolicy Bypass -File scripts\watch_training.ps1 [-Log train_p1]
# The batch counter updates at each checkpoint (every ~5 min); epoch results appear as they finish.
param([string]$Log = "train_p1", [int]$EverySeconds = 10)

$path = Join-Path $PSScriptRoot "..\results\training\logs\$Log.log"
$shown = 0
while ($true) {
    if (-not (Test-Path $path)) { Write-Host "waiting for $path ..."; Start-Sleep $EverySeconds; continue }
    $lines = Get-Content $path
    # print every new epoch / test / error line once
    $new = $lines[$shown..($lines.Count - 1)] | Where-Object { $_ -match "\] epoch \d+/\d+ \||TEST|resumed|already done|Traceback|Error|All requested|Paused" }
    $new | ForEach-Object {
        $c = if ($_ -match "Traceback|Error") { "Red" } elseif ($_ -match "TEST|\*best\*") { "Green" } else { "Gray" }
        Write-Host $_ -ForegroundColor $c
    }
    $shown = $lines.Count
    $ck = $lines | Select-String "\[(.+?)\] epoch (\d+) batch (\d+)/(\d+) \| loss ([\d.]+)" | Select-Object -Last 1
    if ($ck) {
        $g = $ck.Matches[0].Groups
        $pct = [math]::Round(100 * [int]$g[3].Value / [int]$g[4].Value, 1)
        Write-Progress -Activity "Pixel-Artifacts training [$($g[1].Value)]" `
            -Status "epoch $($g[2].Value): batch $($g[3].Value)/$($g[4].Value) ($pct %)  train loss $($g[5].Value)" -PercentComplete $pct
    }
    if ($lines -match "All requested runs done|Traceback") { [console]::Beep(880, 400); [console]::Beep(1175, 600); break }
    Start-Sleep $EverySeconds
}
