# Unattended queue for one GPU "slot": wait for a running trainer to exit, then run each
# train_heads.py argument set in turn. A crashed command is retried (runs resume from their
# checkpoints), so a night of training survives transient failures.
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy Bypass -File scripts\run_queue.ps1 -WaitPid 123 -Name slotA -Jobs "--group p1; --group p3 --part 1/2 --slot 1/3"'
param([int]$WaitPid = 0, [string]$Name = "slot", [string]$Jobs, [int]$Retries = 3)   # Jobs: ";"-separated

$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "results\training\logs"
$qlog = Join-Path $logDir "queue_$Name.log"
function Note($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Add-Content $qlog }

if ($WaitPid) {
    Note "waiting for PID $WaitPid"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 30 }
}
foreach ($job in ($Jobs -split ";" | % { $_.Trim() } | ? { $_ })) {
    $tag = ($job -replace '--', '' -replace '[ /]', '_')
    $log = Join-Path $logDir "train_$($Name)_$tag.log"
    for ($try = 1; $try -le $Retries; $try++) {
        Note "start ($try/$Retries): $job -> $log"
        $p = Start-Process -FilePath $py -ArgumentList "-u scripts/train_heads.py $job" -WorkingDirectory $repo `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru -Wait
        Note "exit code $($p.ExitCode): $job"
        if ($p.ExitCode -eq 0) { break }
        Start-Sleep 60
    }
}
Note "queue finished"
