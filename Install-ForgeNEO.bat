@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title ForgeNEO Anima セットアップ
set "FORGENEO_PARENT=%~dp0"
set "FORGENEO_ROOT=%~dp0ForgeNEO-Anima"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "EXPECTED=e4d6b67c5507eefd0f42b47d3c1d54eb29ffef46c4a2949573b7539f1e4a084b"
echo ForgeNEO Anima セットアップ
echo 保存先 "%FORGENEO_ROOT%"
if exist "%FORGENEO_ROOT%\.setup-v1" goto install
powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; if(-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'){throw 'Windows x64 is required'}; $root=$env:FORGENEO_ROOT; if(Test-Path -LiteralPath $root){throw 'ForgeNEO-Anima already exists without its setup marker. Move the BAT into an empty folder.'}; $stage=Join-Path $env:FORGENEO_PARENT ('.ForgeNEO-stage-'+[guid]::NewGuid()); $zip=$stage+'.zip'; $url='https://github.com/asugonomi/ForgeNEO-Anima-Setup/releases/download/v1.0.0/setup.zip'; Write-Host 'GitHub Releases からセットアップを取得中...'; try { & curl.exe -fLsS --retry 3 --connect-timeout 20 --proto '=https' --proto-redir '=https' -o $zip $url; if($LASTEXITCODE){throw 'setup.zip download failed'}; if((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash -ne $env:EXPECTED){throw 'setup.zip SHA-256 mismatch'}; [void][IO.Directory]::CreateDirectory($stage); & tar.exe -xf $zip -C $stage; if($LASTEXITCODE){throw 'setup.zip extraction failed'}; [IO.File]::WriteAllText((Join-Path $stage '.setup-v1'),'1'); [IO.Directory]::Move($stage,$root) } finally { if(Test-Path -LiteralPath $zip){Remove-Item -LiteralPath $zip -Force} }"
if errorlevel 1 goto failed
:install
pushd "%FORGENEO_ROOT%" || goto failed
"python\python.exe" -E -s -X utf8 -u manager.py install
set "RESULT=%ERRORLEVEL%"
popd
goto done
:failed
set "RESULT=1"
:done
if not "%RESULT%"=="0" echo セットアップを完了できませんでした。上のエラーをご確認ください。
if /i not "%~1"=="--no-pause" pause
exit /b %RESULT%
