<#
run_ftsrc_variant.ps1 — fine-tuning source variant: ft_src = mediapipe vs both
===============================================================================
The base model is fixed at both (joint two-domain training); the fine-tuning stage uses either:
    mediapipe  deployment scenario (at calibration the user only has a phone)
    both       academic upper bound (assumes the person also has multi-camera data; unavailable in practice)
Both run within **the same execution**, so they can be subtracted directly and tested pairwise.

🔴 Why this script has no parameters:
   On 2026-10-03, starting chain_finetune.ps1 with `-FtSrc mediapipe both` failed;
   the process never started. The reason: when PowerShell is called with `-File`,
   **a space-separated array binds only the first value** (it needs commas: `-FtSrc mediapipe,both`).
   On top of that, the log file still held the previous run's output, so it looked as if it were running.
   So this script hard-codes its parameters and uses fresh log paths, leaving no room to repeat that mistake.

Usage:
    .\experiments\run_ftsrc_variant.ps1
To stop:
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*dualdomain_finetune*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    (stop this powershell first, otherwise the retry loop restarts it)
#>
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$CodeDir = Split-Path -Parent $PSScriptRoot
Set-Location $CodeDir

$env:CUDA_CACHE_MAXSIZE = '4294967296'
$env:TF_FORCE_GPU_ALLOW_GROWTH = 'true'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Python = 'python'
$o = Join-Path $env:TEMP 'ftv2_out.txt'      # fresh path, not reusing ft_out.txt
$e = Join-Path $env:TEMP 'ftv2_err.txt'
$log = Join-Path $env:TEMP 'ftv2_chain.txt'

$pyArgs = @(
    'experiments/dualdomain_finetune.py',
    '--prefix', 'dualdomain_finetune_v2',
    '--resume',
    '--arms', 'both',
    '--ft-src', 'mediapipe', 'both',
    '--seeds', '42', '1', '2'
)

Add-Content $log "===== Started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" -Encoding utf8
Add-Content $log ("args: " + ($pyArgs -join ' ')) -Encoding utf8

for ($i = 1; $i -le 6; $i++) {
    $m = "----- Attempt $i $(Get-Date -Format 'HH:mm:ss') -----"
    Write-Host $m; Add-Content $log $m -Encoding utf8
    $p = Start-Process -FilePath $Python -ArgumentList $pyArgs -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $o -RedirectStandardError $e
    if ($p.ExitCode -eq 0) {
        $m = "===== Done $(Get-Date -Format 'HH:mm:ss') ====="
        Write-Host $m -ForegroundColor Green; Add-Content $log $m -Encoding utf8
        exit 0
    }
    $txt = ''
    if (Test-Path $e) { $txt = Get-Content $e -Raw -Encoding utf8 }
    if ($txt -match 'CUDA_ERROR|cuModuleGetFunction|cuLaunchKernel|InternalError') {
        $m = "  Sporadic CUDA error (rc=$($p.ExitCode)); resuming in 30 seconds"
        Write-Host $m -ForegroundColor Yellow; Add-Content $log $m -Encoding utf8
        Start-Sleep -Seconds 30
    } else {
        $m = "  Non-CUDA error (rc=$($p.ExitCode)); stopping. See $e"
        Write-Host $m -ForegroundColor Red; Add-Content $log $m -Encoding utf8
        exit $p.ExitCode
    }
}
Add-Content $log 'Still not finished after 6 attempts' -Encoding utf8
exit 1
