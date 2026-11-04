# fp-v4.2 pipeline after the fine-tune and the ViT-L/14 pass already running (DECISION_LOG D-22).
# Every step is resumable and retried, so rerunning this script after a crash continues the work.
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy Bypass -File scripts\run_v42.ps1 -FinetunePid 1 -SemanticPid 2'
param([int]$FinetunePid = 0, [int]$SemanticPid = 0, [int]$Retries = 3)

$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "results\training\logs"
$qlog = Join-Path $logDir "queue_v42.log"
function Note($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Add-Content $qlog }
function WaitFor($id) {
    if ($id) { Note "waiting for PID $id"; while (Get-Process -Id $id -ErrorAction SilentlyContinue) { Start-Sleep 30 } }
}
function Step($tag, $cmd) {
    $log = Join-Path $logDir "train_v42_$tag.log"
    for ($try = 1; $try -le $Retries; $try++) {
        Note "start ($try/$Retries): $cmd"
        $p = Start-Process -FilePath $py -ArgumentList "-u $cmd" -WorkingDirectory $repo `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru -Wait
        Note "exit code $($p.ExitCode): $cmd"
        if ($p.ExitCode -eq 0) { return }
        Start-Sleep 60
    }
    Note "giving up: $cmd"; exit 1
}

WaitFor $FinetunePid
Step "finetune_spatial_v2_check" "scripts/finetune_spatial_v2.py"          # no-op if DONE.json exists
Step "spatial_raw" "scripts/build_store_v42.py --pillars spatial --threads 6"
WaitFor $SemanticPid
Step "semantic_raw_check" "scripts/build_store_v42.py --pillars semantic --threads 8"
Step "assemble" "scripts/build_store_v42.py --pillars --assemble"
Step "head" "scripts/train_heads.py --group p1 --store results/training/feature_store_v42 --runs results/training/runs_v42"
Note "fp-v4.2 pipeline finished"
