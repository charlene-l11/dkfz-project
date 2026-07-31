@echo off
setlocal
cd /d "%~dp0"
python -m median_pipeline run --config configs\manifold_settings.yaml
exit /b %ERRORLEVEL%
