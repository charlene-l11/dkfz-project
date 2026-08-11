@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_windows.bat "C:\path\to\scenario\manifold_settings.yaml"
  exit /b 2
)
python -m median_pipeline run --config "%~1"
exit /b %ERRORLEVEL%
