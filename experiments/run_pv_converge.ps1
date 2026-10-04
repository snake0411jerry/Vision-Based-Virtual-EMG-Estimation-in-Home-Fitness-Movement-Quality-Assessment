<#
run_pv_converge.ps1 — converged rerun of slide 22's “0.830 after fine-tuning”
===============================================================================
See the header of experiments/pipeline_variance_converge.py.
The same LOSO base model branches into two paths:
  (a) original protocol: 20 epochs + early stopping on the test tail (should reproduce ≈0.830)
  (b) 150 epochs without early stopping, recording the curve at every epoch
3 seeds (42/1/2, the first three of the original protocol's 5 seeds), about 2.2 hours.

Parameters are hard-coded and the log uses its own path: on 2026-10-03, when called with parameters, PowerShell passing an array via -File
bound only the first value, so the process never started; and reading a leftover log from the previous run made it look as if it were running.

To stop: stop this powershell first, then python (otherwise the retry loop restarts it)
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_pv_converge*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*pipeline_variance_converge*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
#>
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$CodeDir = Split-Path -Parent $PSScriptRoot
Set-Location $CodeDir

$env:CUDA_CACHE_MAXSIZE = '4294967296'
$env:TF_FORCE_GPU_ALLOW_GROWTH = 'true'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Python = 'python'
$o = Join-Path $env:TEMP 'pvc_out.txt'
$e = Join-Path $env:TEMP 'pvc_err.txt'
$log = Join-Path $env:TEMP 'pvc_chain.txt'

$pyArgs = @(
    'experiments/pipeline_variance_converge.py',
    '--prefix', 'pv_converge',
    '--resume',
    '--seeds', '42', '1', '2',
    '--ft-epochs', '150'
)

Add-Content $log "===== Started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" -Encoding utf8
Add-Content $log ('args: ' + ($pyArgs -join ' ')) -Encoding utf8

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
