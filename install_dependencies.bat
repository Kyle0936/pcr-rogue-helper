@echo off
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt
if %ERRORLEVEL% EQU 0 exit /b 0
python -m pip install -r requirements.txt
