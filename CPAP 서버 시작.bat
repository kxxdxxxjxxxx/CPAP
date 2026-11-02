@echo off
rem CPAP analysis server - one-click launcher
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
title CPAP Server

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating Python virtual environment .venv ...
    where py >nul 2>nul && (py -3.11 -m venv .venv || py -3 -m venv .venv) || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [error] Python not found. Install Python 3.11 from https://www.python.org and run again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "server\launcher.py"
echo.
pause
