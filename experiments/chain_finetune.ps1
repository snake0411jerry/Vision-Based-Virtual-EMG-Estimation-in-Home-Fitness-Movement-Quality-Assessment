<#
chain_finetune.ps1 — wait for the 5-seed loso_dualdomain run to finish, then run personalized fine-tuning
===============================================================================
Both experiments saturate the GPU and slow each other down when run together (and PTX JIT on this
machine's sm_120 is unstable to begin with), so this script chains them: it polls until the previous
process is gone, then starts the next one.

Includes retry: this machine's RTX 5060 Laptop (compute capability 12.0) with TF 2.10.0
has no precompiled sm_120 kernels and occasionally crashes with cuModuleGetFunction -> CUDA_ERROR_UNKNOWN.
dualdomain_finetune.py has --resume, so a retry continues from the unfinished (arm, seed) pairs.

Usage:
    .\experiments\chain_finetune.ps1
    .\experiments\chain_finetune.ps1 -SkipWait      # don't wait, start immediately
#>
[CmdletBinding()]
param(
    [int[]]$Seeds = @(42, 1, 2),
    [string[]]$Arms = @('both', 'mp_only'),
    [string[]]$FtSrc = @('mediapipe'),
    [string]$Prefix = 'dualdomain_finetune',
    [int]$MaxTry = 6,
    [string]$Python = 'python',
    [switch]$SkipWait
)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$CodeDir = Split-Path -Parent $PSScriptRoot
Set-Location $CodeDir

$env:CUDA_CACHE_MAXSIZE = '4294967296'
$env:TF_FORCE_GPU_ALLOW_GROWTH = 'true'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$log = Join-Path $env:TEMP 'ft_chain.txt'
$o = Join-Path $env:TEMP 'ft_out.txt'
$e = Join-Path $env:TEMP 'ft_err.txt'

function Running([string]$pat) {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like $pat }).Count -gt 0
}

if (-not $SkipWait) {
    $msg = "Waiting for loso_dualdomain to finish... $(Get-Date -Format 'HH:mm:ss')"
    Write-Host $msg; Add-Content $log $msg -Encoding utf8
    while (Running '*loso_dualdomain.py*') { Start-Sleep -Seconds 60 }
    $msg = "Previous experiment finished at $(Get-Date -Format 'HH:mm:ss'); starting fine-tuning in 30 seconds"
    Write-Host $msg; Add-Content $log $msg -Encoding utf8
    Start-Sleep -Seconds 30
}

$pyArgs = @('experiments/dualdomain_finetune.py', '--resume',
            '--prefix', $Prefix, '--seeds') +
          ($Seeds | ForEach-Object { "$_" }) +
          @('--arms') + $Arms + @('--ft-src') + $FtSrc

for ($i = 1; $i -le $MaxTry; $i++) {
    $msg = "===== Fine-tuning attempt $i  $(Get-Date -Format 'HH:mm:ss') ====="
    Write-Host $msg; Add-Content $log $msg -Encoding utf8
    $p = Start-Process -FilePath $Python -ArgumentList $pyArgs -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $o -RedirectStandardError $e
    if ($p.ExitCode -eq 0) {
        $msg = "===== Fine-tuning done  $(Get-Date -Format 'HH:mm:ss') ====="
        Write-Host $msg -ForegroundColor Green; Add-Content $log $msg -Encoding utf8
        exit 0
    }
    $txt = ''
    if (Test-Path $e) { $txt = Get-Content $e -Raw -Encoding utf8 }
    if ($txt -match 'CUDA_ERROR|cuModuleGetFunction|cuLaunchKernel|InternalError') {
        $msg = "  Sporadic CUDA error (rc=$($p.ExitCode)); resuming in 30 seconds"
        Write-Host $msg -ForegroundColor Yellow; Add-Content $log $msg -Encoding utf8
        Start-Sleep -Seconds 30
    } else {
        $msg = "  Non-CUDA error (rc=$($p.ExitCode)) — retrying won't help; stopping. See $e"
        Write-Host $msg -ForegroundColor Red; Add-Content $log $msg -Encoding utf8
        exit $p.ExitCode
    }
}
Add-Content $log "Still not finished after $MaxTry attempts" -Encoding utf8
exit 1
