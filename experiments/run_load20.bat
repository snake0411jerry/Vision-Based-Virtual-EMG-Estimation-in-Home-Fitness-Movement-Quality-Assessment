@echo off
REM English summary: Windows launcher for run_load20.py.
rem Plan 3 one-click launcher. Double-click to run. Edit ENV_NAME if your conda env is not pose_ai.
set ENV_NAME=pose_ai
chcp 65001 >nul
cd /d "%~dp0.."
where conda >nul 2>nul
if %errorlevel%==0 (
    call conda activate %ENV_NAME%
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" %ENV_NAME%
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" %ENV_NAME%
)
python experiments\run_load20.py %*
echo.
echo Finished. Press any key to close.
pause >nul
