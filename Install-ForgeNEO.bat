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
set "EXPECTED=7298e8db9e7ab13ada7e03794ccb8aad927465eb9d521290c48c8be44327d9c8"
echo ForgeNEO Anima セットアップ
echo 保存先 "%FORGENEO_ROOT%"
if exist "%FORGENEO_ROOT%\.setup-v1.2.0" goto install
powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; if(-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'){throw 'Windows x64 is required'}; $root=$env:FORGENEO_ROOT; if((Test-Path -LiteralPath $root) -and !(Test-Path -LiteralPath (Join-Path $root '.setup-v1'))){throw 'ForgeNEO-Anima already exists without its setup marker. Move the BAT into an empty folder.'}; $stage=Join-Path $env:FORGENEO_PARENT ('.ForgeNEO-stage-'+[guid]::NewGuid()); $zip=$stage+'.zip'; $url='https://github.com/asugonomi/ForgeNEO-Anima-Setup/releases/download/v1.2.0/setup.zip'; Write-Host 'GitHub Releases からセットアップを取得中...'; try { & curl.exe -fLsS --retry 3 --connect-timeout 20 --proto '=https' --proto-redir '=https' -o $zip $url; if($LASTEXITCODE){throw 'setup.zip download failed'}; $alg=[Security.Cryptography.SHA256]::Create(); $stream=[IO.File]::OpenRead($zip); try { $hash=[BitConverter]::ToString($alg.ComputeHash($stream)).Replace('-','') } finally { $stream.Dispose(); $alg.Dispose() }; if($hash -ne $env:EXPECTED){throw 'setup.zip SHA-256 mismatch'}; [void][IO.Directory]::CreateDirectory($stage); & tar.exe -xf $zip -C $stage; if($LASTEXITCODE){throw 'setup.zip extraction failed'}; [IO.File]::WriteAllText((Join-Path $stage '.setup-v1'),'1'); if(Test-Path -LiteralPath $root){ & (Join-Path $root 'python/python.exe') -E -s -X utf8 (Join-Path $stage 'upgrade_setup.py') $root; if($LASTEXITCODE){throw 'Existing setup upgrade failed'} } else { [IO.File]::WriteAllText((Join-Path $stage '.setup-v1.2.0'),'1'); [IO.Directory]::Move($stage,$root) } } finally { if([IO.Directory]::Exists($stage)){[IO.Directory]::Delete($stage,$true)}; if(Test-Path -LiteralPath $zip){Remove-Item -LiteralPath $zip -Force} }"
if errorlevel 1 goto failed
:install
pushd "%FORGENEO_ROOT%" || goto failed
set "CPU_ARG="
if /i "%~1"=="--cpu" set "CPU_ARG=--cpu"
if /i "%~2"=="--cpu" set "CPU_ARG=--cpu"
"python\python.exe" -E -s -X utf8 -u manager.py install %CPU_ARG%
set "RESULT=%ERRORLEVEL%"
popd
goto done
:failed
set "RESULT=1"
:done
if not "%RESULT%"=="0" echo セットアップを完了できませんでした。上のエラーをご確認ください。
if /i not "%~1"=="--no-pause" if /i not "%~2"=="--no-pause" pause
exit /b %RESULT%
