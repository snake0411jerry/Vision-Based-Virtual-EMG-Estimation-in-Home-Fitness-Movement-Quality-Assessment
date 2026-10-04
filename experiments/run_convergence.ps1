<#
run_convergence.ps1 — fine-tuning convergence experiment
===============================================================================
Addresses the two problems found on 2026-10-03 (see MEDIAPIPE_EXPERIMENT.md §9.1d):

  Problem 1  Fine-tuning had not converged at all. TailProbe showed 93–100% of folds
             still rising at epoch 20, so every “r after fine-tuning” is a lower bound.
             FT_EPOCHS=20 was inherited from phase2_finetune_loso.py; it is not a convergence point.

  Problem 2  The advantage of ft_src=both is completely confounded with “twice the gradient steps” —
             both has 6057 fine-tuning windows, mediapipe has 3029, exactly twice as many.
             When neither group has converged, taking twice the steps is better by itself.

Running to convergence solves both: once the curves flatten, the step-count difference is no longer a confounder.

Settings: ft-epochs 100, both ft_src variants, 3 seeds, per-epoch curves saved.

🔴 The curves are for **diagnosis only**: never use them to pick an epoch (that would be tuning on the test set).
   The correct use is to see where they flatten, fix that epoch count in advance as a constant and run again;
   if all curves are already flat at 100 epochs, then the choice of “100” is independent of test performance
   and citing the 100-epoch numbers directly is clean.

Cost: 3 seeds × (base model ~38 min + 2 ft_src × 100 epochs of fine-tuning and probing)
      ≈ 2.5–3 hours.

Usage:
    .\experiments\run_convergence.ps1
To stop: stop this powershell first, then python (otherwise the retry loop restarts it):
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_convergence*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*dualdomain_finetune*' } |
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
$o = Join-Path $env:TEMP 'conv_out.txt'
$e = Join-Path $env:TEMP 'conv_err.txt'
$log = Join-Path $env:TEMP 'conv_chain.txt'

# Parameters are hard-coded: on 2026-10-03, when called with parameters, PowerShell passing an array via -File
# bound only the first value (it needs commas), so the process never started and nobody noticed.
$pyArgs = @(
    'experiments/dualdomain_finetune.py',
    '--prefix', 'dualdomain_converge',
    '--resume',
    '--arms', 'both',
    '--ft-src', 'mediapipe', 'both',
    '--seeds', '42', '1', '2',
    '--ft-epochs', '100',
    '--save-curve'
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
