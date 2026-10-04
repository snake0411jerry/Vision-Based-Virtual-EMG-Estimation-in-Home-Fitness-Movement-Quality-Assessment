<#
run_mediapipe.ps1 — run the MediaPipe vs OpenCap LOSO comparison, resuming automatically after crashes
===============================================================================
Why this wrapper is needed:
  This machine is an RTX 5060 Laptop (compute capability 12.0) with TensorFlow 2.10.0.
  TF says so itself at startup:
    "TensorFlow was not built with CUDA kernel binaries compatible with
     compute capability 12.0. CUDA kernels will be jit-compiled from PTX"
  Every kernel has to be compiled on the fly from PTX by the driver. Two consequences:
    1. GPU utilization is only ~27% (bottlenecked by JIT and the batch=32 Python loop)
    2. The whole process occasionally crashes:
       'cuModuleGetFunction(...)' failed with 'CUDA_ERROR_UNKNOWN'
       This happened once at 10:55 on 2026-10-02 (after 23 minutes of running)
  So an outer retry loop is added, together with loso_multiseed.py --resume:
  completed (arm, seed) pairs are skipped, so every retry makes progress.

⚠️ The resume granularity of --resume is “one fold-set (10 folds, about 20 minutes)”.
   Crashing in the middle of a fold-set loses that set.

⚠️ The execution order puts seeds in the outer loop (loso_multiseed.py --order seed, the default).
   This means **whenever you stop, every arm has the same number of seeds**, so partial results can still be used for paired tests.
   With arms in the outer loop, stopping midway gives asymmetric results like “A has 3 seeds, B only 1”,
   which is wasted work — it nearly happened on 2026-10-02.

Usage (in PowerShell):
    .\experiments\run_mediapipe.ps1              # only the two headline arms (mp16 / oc16)
    .\experiments\run_mediapipe.ps1 -Arms all    # all eight arms
    .\experiments\run_mediapipe.ps1 -Arms all -Seeds 42,1,2

To stop, use .\experiments\stop_mediapipe.ps1 (do not just kill python;
the outer loop will treat it as a crash and restart).
#>
[CmdletBinding()]
param(
    [ValidateSet('headline', 'all')]
    [string]$Arms = 'headline',
    [int[]]$Seeds = @(42, 1, 2),
    [int]$MaxTry = 8,
    [string]$Python = 'python',
    [string]$LogPath = '',
    # Only print the arguments that would be passed to python, then exit without running.
    # Used to confirm that arm strings containing `~` and `=` are passed through intact (the easiest place for the shell to mangle them).
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

# the parent of experiments/ is Code/
$CodeDir = Split-Path -Parent $PSScriptRoot
Set-Location $CodeDir

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python not found: $Python  (use -Python to specify the correct path)"
}

# ---- JIT cache: sm_120 has no precompiled kernels; the cache must be large enough to avoid repeated recompilation ----
$env:CUDA_CACHE_MAXSIZE      = '4294967296'   # 4 GB
$env:CUDA_CACHE_DISABLE      = '0'
$env:TF_FORCE_GPU_ALLOW_GROWTH = 'true'       # don't grab 5.5/8 GB up front
$env:PYTHONUNBUFFERED        = '1'            # otherwise print is block-buffered and live progress is invisible
$env:PYTHONIOENCODING        = 'utf-8'

$R8 = 'Shoulder_Y_norm,Knee_Y_norm,Ankle_Y_norm,Toe_Y_norm,Knee_X_norm,Knee_Angle_norm,Trunk_Lean_Angle_norm,Knee_Ankle_Ratio_norm'
$R7 = 'Shoulder_Y_norm,Knee_Y_norm,Ankle_Y_norm,Toe_Y_norm,Knee_X_norm,Knee_Angle_norm,Knee_Ankle_Ratio_norm'

# headline: compares only one thing, “changing the skeleton source”, 16 dims vs 16 dims
$Headline = @(
    'mp16=Combined_mediapipe',
    'oc16=Combined'
)
# all eight arms. --resume skips completed ones, so 'all' can be used directly to resume.
$All = $Headline + @(
    "mp8=Combined_mediapipe~$R8",
    "oc8=Combined~$R8",
    "mp7=Combined_mediapipe~$R7",
    "oc7=Combined~$R7",
    "mp8tf=Combined_mediapipe_trunkfix~$R8",
    "oc8tf=Combined_trunkfix~$R8"
)

if ($Arms -eq 'all') { $Sel = $All } else { $Sel = $Headline }

if (-not $LogPath) {
    $LogPath = Join-Path $env:TEMP 'mp_headline.txt'
}
$tmpOut = Join-Path $env:TEMP 'mp_run_out.txt'
$tmpErr = Join-Path $env:TEMP 'mp_run_err.txt'

Write-Host "Code dir     : $CodeDir"
Write-Host "Summary log  : $LogPath   (written only after each attempt ends)"
Write-Host "Live progress: $tmpOut   <- look here for progress while running" -ForegroundColor Cyan
Write-Host "   Get-Content `"$tmpOut`" -Tail 10 -Wait"
Write-Host "arm          : $($Sel -join '  ')"
Write-Host "seed         : $($Seeds -join ', ')"
Write-Host ""

$pyArgs = @(
    'experiments/loso_multiseed.py',
    '--prefix', 'loso_mediapipe',
    '--resume',
    '--seeds'
) + ($Seeds | ForEach-Object { "$_" }) + @('--arms') + $Sel

if ($DryRun) {
    Write-Host "Would run:" -ForegroundColor Cyan
    Write-Host "  $Python"
    for ($k = 0; $k -lt $pyArgs.Count; $k++) {
        Write-Host ("    [{0,2}] {1}" -f $k, $pyArgs[$k])
    }
    Write-Host ""
    Write-Host "Environment variables:" -ForegroundColor Cyan
    foreach ($v in @('CUDA_CACHE_MAXSIZE', 'CUDA_CACHE_DISABLE',
                     'TF_FORCE_GPU_ALLOW_GROWTH', 'PYTHONUNBUFFERED')) {
        Write-Host ("    {0} = {1}" -f $v, (Get-Item "env:$v").Value)
    }
    exit 0
}

for ($i = 1; $i -le $MaxTry; $i++) {
    $stamp = Get-Date -Format 'HH:mm:ss'
    $banner = "===== Attempt $i  $stamp ====="
    Write-Host $banner
    Add-Content -Path $LogPath -Value $banner -Encoding utf8

    # Start-Process redirects stdout/stderr separately.
    # In PS 5.1, `2>&1` on a native exe wraps every line in an ErrorRecord and sets $? to false,
    # even when the exit code is 0 — so no redirection operators here.
    #
    # ⚠️ $tmpOut/$tmpErr are appended to $LogPath only **after** python exits,
    #    so $LogPath does not contain the current attempt while it is running.
    #    For live progress look at $tmpOut directly (Start-Process redirection is written in real time,
    #    and the script sets PYTHONUNBUFFERED=1, so it won't stall in block buffering).
    $p = Start-Process -FilePath $Python -ArgumentList $pyArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    $rc = $p.ExitCode

    foreach ($f in @($tmpOut, $tmpErr)) {
        if (Test-Path $f) {
            Get-Content $f -Encoding utf8 | Add-Content -Path $LogPath -Encoding utf8
        }
    }
    if (Test-Path $tmpOut) { Get-Content $tmpOut -Encoding utf8 | Where-Object { $_ -match '^\[\d+/\d+\]|to run|already done|Summary across seeds' } | ForEach-Object { Write-Host $_ } }

    if ($rc -eq 0) {
        $done = "===== Done (attempt $i)  $(Get-Date -Format 'HH:mm:ss') ====="
        Write-Host $done -ForegroundColor Green
        Add-Content -Path $LogPath -Value $done -Encoding utf8
        exit 0
    }

    # sporadic CUDA errors are worth retrying; real program errors won't get better with retries
    $errText = ''
    if (Test-Path $tmpErr) { $errText = (Get-Content $tmpErr -Raw -Encoding utf8) }
    if ($errText -match 'CUDA_ERROR|cuModuleGetFunction|cuLaunchKernel|InternalError') {
        $msg = "  Sporadic CUDA error (rc=$rc); resuming in 30 seconds"
        Write-Host $msg -ForegroundColor Yellow
        Add-Content -Path $LogPath -Value $msg -Encoding utf8
        Start-Sleep -Seconds 30
    } else {
        $msg = "  Non-CUDA error (rc=$rc) — retrying won't help; stopping. See $LogPath"
        Write-Host $msg -ForegroundColor Red
        Add-Content -Path $LogPath -Value $msg -Encoding utf8
        exit $rc
    }
}

$msg = "Still not finished after $MaxTry attempts; stopping. See $LogPath"
Write-Host $msg -ForegroundColor Red
Add-Content -Path $LogPath -Value $msg -Encoding utf8
exit 1
