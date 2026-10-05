<#
run_base_converge.ps1 — base model mp_only vs both, converged version
===============================================================================
At FT_EPOCHS=20, §9.1b found that “personalization washes out the benefit of joint two-domain training”
(after fine-tuning both − mp_only = +0.003, p=0.28). But §9.1e showed that 20 epochs is
far from converged, so this conclusion has to be measured again after convergence.

Settings: two base models, both / mp_only; fine-tuning uses phone data only (ft_src=mediapipe,
the deployment scenario), 150 epochs without early stopping (same as pv_converge), 3 seeds, per-epoch curves saved.
Both run within **the same execution**, so they can be subtracted pairwise directly.

🔴 The curves are for diagnosis only; never use them to pick an epoch (see the header of run_convergence.ps1).

Cost: 3 seeds × (both base ~38 min + mp_only base ~19 min + 2 × 150 epochs of fine-tuning)
      ≈ 4–6 hours (6h20m on the development machine).

Parameters are hard-coded and the log uses its own path (see the header of run_ftsrc_variant.ps1 for why).

Usage:
    .\experiments\run_base_converge.ps1
To stop: stop this powershell first, then python (otherwise the retry loop restarts it):
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_base_converge*' } |
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
$o = Join-Path $env:TEMP 'bconv_out.txt'
$e = Join-Path $env:TEMP 'bconv_err.txt'
$log = Join-Path $env:TEMP 'bconv_chain.txt'

$pyArgs = @(
    'experiments/dualdomain_finetune.py',
    '--prefix', 'dualdomain_base_converge',
    '--resume',
    '--arms', 'both', 'mp_only',
    '--ft-src', 'mediapipe',
    '--seeds', '42', '1', '2',
    '--ft-epochs', '150',
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
