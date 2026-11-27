@echo off
setlocal
cd /d "%~dp0"
set "PH3PY="
if exist "..\..\venv\Scripts\python.exe" set "PH3PY=..\..\venv\Scripts\python.exe"
if not defined PH3PY if exist "..\..\system\python\python.exe" set "PH3PY=..\..\system\python\python.exe"
if not defined PH3PY if exist "..\..\..\system\python\python.exe" set "PH3PY=..\..\..\system\python\python.exe"
if defined PH3PY (
  "%PH3PY%" tools\setup_native.py %*
) else (
  python tools\setup_native.py %*
)
if errorlevel 1 echo Setup failed. No Forge packages were changed. See the message above.
pause
