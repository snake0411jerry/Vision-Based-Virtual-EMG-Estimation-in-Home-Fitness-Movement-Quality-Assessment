# English summary: Chains load_curve_20.py and load_dropout_20.py workers (PowerShell).
# 依序跑：加負重學習曲線（load_curve_20.py）→ 負重 dropout（load_dropout_20.py），各 $Workers 個 worker（預設 4）。
# 兩個腳本都可中斷續跑；重跑此檔會跳過已完成的工作。
# 用法（隱藏視窗、與 Claude 工作階段分離）：
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','experiments\run_load_extra.ps1'
param([int]$Workers = 4)   # 5 個 worker 每個約 10 GB 記憶體，47 GB 的筆電會用滿而卡頓，預設改 4 個
$ErrorActionPreference = 'Stop'
$py = 'C:\Users\jeffe\anaconda3\envs\pose_ai\python.exe'
$code = Split-Path -Parent $PSScriptRoot
Set-Location $code
$env:PYTHONIOENCODING = 'utf-8'
foreach ($pair in @(@('load_curve_20', 'load_curve'), @('load_dropout_20', 'load_dropout'))) {
    $script, $dir = $pair
    $out = Join-Path $code "results\$dir"
    New-Item -ItemType Directory -Force $out | Out-Null
    Remove-Item (Join-Path $out 'run_w*.log'), (Join-Path $out 'err_w*.log') -ErrorAction SilentlyContinue
    $procs = foreach ($k in 0..($Workers - 1)) {
        Start-Process -FilePath $py -WorkingDirectory $code -WindowStyle Hidden -PassThru `
            -ArgumentList "experiments\$script.py", '--worker', $k, '--n-workers', $Workers `
            -RedirectStandardOutput (Join-Path $out "run_w$k.log") -RedirectStandardError (Join-Path $out "err_w$k.log")
    }
    $procs | ForEach-Object { try { $_.PriorityClass = 'BelowNormal' } catch {} }   # 讓前景操作優先，電腦不卡
    $procs | Wait-Process
    Add-Content -Path (Join-Path $out 'chain.log') -Value "$(Get-Date -Format 'HH:mm') $script 全部 worker 結束" -Encoding utf8
}
