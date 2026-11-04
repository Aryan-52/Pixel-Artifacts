# Live progress bar for feature extraction (reads the extraction logs; does not touch the job).
#   powershell -ExecutionPolicy Bypass -File scripts\watch_extraction.ps1 [-Split train]
param([string]$Split = "train", [int]$EverySeconds = 5)

$log = Join-Path $PSScriptRoot "..\results\training\logs\extract_$Split.log"
while ($true) {
    if (-not (Test-Path $log)) { Write-Host "waiting for $log ..."; Start-Sleep $EverySeconds; continue }
    $line = Select-String -Path $log -Pattern "shard (\d+)/(\d+) saved" | Select-Object -Last 1
    if ($line) {
        $m = $line.Matches[0]
        $done, $total = [int]$m.Groups[1].Value, [int]$m.Groups[2].Value
        $pct = [math]::Round(100 * $done / $total, 1)
        $info = ($line.Line -split '\|')[2..4] -join '|'          # img/s | ETA | sha mismatches
        Write-Progress -Activity "Pixel-Artifacts feature extraction ($Split)" `
            -Status "shard $done / $total  ($pct %)  $info" -PercentComplete $pct
        if ($done -ge $total) {
            Write-Progress -Activity "Pixel-Artifacts feature extraction ($Split)" -Completed
            Write-Host "`n$Split extraction COMPLETE: $done/$total shards. $info" -ForegroundColor Green
            [console]::Beep(880, 400); [console]::Beep(1175, 600)
            break
        }
    }
    if (Select-String -Path $log -Pattern "Traceback|Error" -Quiet) {
        Write-Host "`nERROR in $log :" -ForegroundColor Red
        Get-Content $log -Tail 15
        break
    }
    Start-Sleep $EverySeconds
}
