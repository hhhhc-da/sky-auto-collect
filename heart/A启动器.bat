@echo off

net session >nul 2>&1

if %errorLevel% neq 0 (
    powershell -Command "Start-Process '%~dpnx0' -Verb RunAs"
    exit /b
)


cd /d "%~dp0"

conda activate data && python ui.py