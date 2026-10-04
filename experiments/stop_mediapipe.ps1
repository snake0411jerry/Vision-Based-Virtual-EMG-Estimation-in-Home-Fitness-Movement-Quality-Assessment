<#
stop_mediapipe.ps1 — pause the MediaPipe experiment cleanly
===============================================================================
Why you can't just kill python:
  run_mediapipe.ps1 is a retry loop (to survive sm_120's sporadic CUDA errors).
  Killing only python makes the outer loop decide “it crashed again” and restart 30 seconds later.
  So the order must be: stop the outer loop first -> then stop python.

🔴 History lesson: the first version used bash `pkill -f`, which under Git Bash **cannot see Windows
   processes at all** and reported “not running” — worse than having no script, because it gave false reassurance.
   Git Bash's procps only sees MSYS's own process table. This version matches CommandLine via Win32_Process.

Completed fold-sets are already written to results/loso_mediapipe_raw.csv (written after each set finishes),
so pausing loses no completed results. The set currently running (up to ~21 minutes) is discarded
and rerun next time with --resume.

Usage (in PowerShell):
    .\experiments\stop_mediapipe.ps1
To resume:
    .\experiments\run_mediapipe.ps1 -Arms all
#>
[CmdletBinding()]
param(
    [string]$Python = 'python'
)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$CodeDir = Split-Path -Parent $PSScriptRoot
Set-Location $CodeDir

function Find-Procs([string]$pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe' OR Name='pwsh.exe' OR Name='bash.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like $pattern -and $_.ProcessId -ne $PID }
}

$steps = @(
    @{ Label = '1) outer retry loop run_mediapipe'; Pattern = '*run_mediapipe*' },
    @{ Label = '2) training process loso_multiseed.py'; Pattern = '*loso_multiseed*' }
)

foreach ($s in $steps) {
    Write-Host $s.Label
    $procs = Find-Procs $s.Pattern
    if ($procs) {
        foreach ($p in $procs) {
            Write-Host ("   stopping PID " + $p.ProcessId)
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host '   not running'
    }
    Start-Sleep -Seconds 2
}

Write-Host '3) check'
Start-Sleep -Seconds 3
$left = @(Find-Procs '*loso_multiseed*') + @(Find-Procs '*run_mediapipe*')
if ($left.Count -gt 0) {
    Write-Host '   🚨 some processes are still alive; handle them manually:' -ForegroundColor Red
    $left | Select-Object ProcessId, Name | Format-Table -AutoSize
    exit 1
}
Write-Host '   clean'
try {
    $g = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    if ($g) { Write-Host "   GPU: $g" }
} catch { }

Write-Host ''
Write-Host '4) results completed so far'
$env:PYTHONIOENCODING = 'utf-8'
$script = @'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
p = 'results/loso_mediapipe_raw.csv'
if not os.path.exists(p):
    print('   no fold-set completed yet')
    raise SystemExit
d = pd.read_csv(p)
n = d.groupby(['arm', 'seed']).size()
print(f'   {len(n)} fold-sets completed:')
for (arm, seed), k in n.items():
    print(f'     {arm:8} seed={seed:<4} {k} folds')
print()
per = d.groupby(['arm', 'seed'])['r_main'].mean()
piv = per.groupby('arm').agg(['mean', 'std', 'count'])
print('   main muscle r per arm (averaged over the 10 people first, then across seeds):')
for arm, r in piv.iterrows():
    sd = '' if pd.isna(r['std']) else f" ± {r['std']:.4f}"
    print(f"     {arm:8} {r['mean']:.4f}{sd}  (n={int(r['count'])} seed)")

# symmetry check: every arm must have the same set of seeds, otherwise paired tests would subtract different seeds
seeds = {a: set(g.index.get_level_values('seed')) for a, g in per.groupby('arm')}
if len(set(map(frozenset, seeds.values()))) > 1:
    print()
    print('   🔴 arms have different seed sets; paired tests must use the intersection first:')
    for a, s in seeds.items():
        print(f'     {a:8} {sorted(s)}')
else:
    print()
    print(f"   all arms have the same seed set ({sorted(next(iter(seeds.values())))}) — they can be compared pairwise directly.")
'@
$tmp = Join-Path $env:TEMP 'mp_stop_report.py'
Set-Content -Path $tmp -Value $script -Encoding utf8
& $Python $tmp

Write-Host ''
Write-Host 'To resume: .\experiments\run_mediapipe.ps1 -Arms all'
