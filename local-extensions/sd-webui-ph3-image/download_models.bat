@echo off
setlocal
cd /d "%~dp0"
set "PH3PY="
if exist "..\..\venv\Scripts\python.exe" set "PH3PY=..\..\venv\Scripts\python.exe"
if not defined PH3PY if exist "..\..\system\python\python.exe" set "PH3PY=..\..\system\python\python.exe"
if not defined PH3PY if exist "..\..\..\system\python\python.exe" set "PH3PY=..\..\..\system\python\python.exe"
if defined PH3PY (
  "%PH3PY%" tools\download_models.py %*
) else (
  python tools\download_models.py %*
)
if errorlevel 1 echo Download failed. Use your Forge Neo Python if huggingface_hub is missing.
pause
