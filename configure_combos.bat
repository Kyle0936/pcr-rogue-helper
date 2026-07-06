@echo off
cd /d "%~dp0"
py -3 pcr_rogue_helper.py --configure-combos --combo-config valid_combos.json
if %ERRORLEVEL% EQU 0 exit /b 0
python pcr_rogue_helper.py --configure-combos --combo-config valid_combos.json
