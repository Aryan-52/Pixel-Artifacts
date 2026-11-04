# Live dashboard of every training run (laptop) + the Kaggle notebook state. Read-only.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_all.ps1
param([int]$EverySeconds = 30)

$repo = Split-Path $PSScriptRoot -Parent
$runsDir = Join-Path $repo "results\training\runs"
$logs = Join-Path $repo "results\training\logs"
$kaggle = Join-Path $repo ".venv\Scripts\kaggle.exe"
$pillars = "spatial", "frequency", "color", "physical", "semantic"
# all 31 runs in priority order: full, singles, leave-one-out, pairs, triples
$combos = @("full") + $pillars
$combos += 0..4 | % { $i = $_; ($pillars | ? { $_ -ne $pillars[$i] }) -join "+" }
for ($a = 0; $a -lt 5; $a++) { for ($b = $a + 1; $b -lt 5; $b++) { $combos += "$($pillars[$a])+$($pillars[$b])" } }
for ($a = 0; $a -lt 5; $a++) { for ($b = $a + 1; $b -lt 5; $b++) { for ($c = $b + 1; $c -lt 5; $c++) {
    $combos += "$($pillars[$a])+$($pillars[$b])+$($pillars[$c])" } } }
$kState = ""; $kTime = [datetime]::MinValue

while ($true) {
    if ((Get-Date) - $kTime -gt [timespan]::FromMinutes(5)) {
        $kState = (& $kaggle kernels status aryandandekar/pixel-artifacts-fp-v4-heads 2>$null | Select-Object -Last 1) -replace '.*status "KernelWorkerStatus\.', '' -replace '"', ''
        $kTime = Get-Date
    }
    $rows = foreach ($c in $combos) {
        $d = Join-Path $runsDir $c
        $status, $ep, $val, $acc, $auc, $real, $fake = "waiting", "", "", "", "", "", ""
        if (Test-Path (Join-Path $d "DONE.json")) {
            $t = Get-Content (Join-Path $d "test_report.json") -Raw | ConvertFrom-Json
            $status = "DONE"; $acc = "{0:N3}" -f $t.overall.accuracy; $auc = "{0:N3}" -f $t.overall.roc_auc
            $real = "{0:N2}" -f $t.overall.real_acc; $fake = "{0:N2}" -f $t.overall.fake_acc
        }
        elseif (Test-Path (Join-Path $d "history.csv")) {
            $h = Import-Csv (Join-Path $d "history.csv"); $last = $h[-1]
            $status = "training"; $ep = $last.epoch; $val = "{0:N3}" -f [double]$last.accuracy
        }
        elseif (Test-Path $d) { $status = "epoch 1" }
        [pscustomobject]@{ Run = $c; Status = $status; Epoch = $ep; ValAcc = $val; TestAcc = $acc; TestAUC = $auc; RealAcc = $real; FakeAcc = $fake }
    }
    $done = ($rows | ? Status -eq "DONE").Count
    Clear-Host
    Write-Host "Pixel-Artifacts fp-v4.0 ablations   $(Get-Date -Format 'HH:mm:ss')   laptop: $done / 31 done   Kaggle notebook: $kState" -ForegroundColor Cyan
    Write-Host "(Kaggle results appear here only after they are downloaded; ValAcc = in-distribution ProGAN val, TestAcc = zero-shot test)`n"
    $rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    Start-Sleep $EverySeconds
}
