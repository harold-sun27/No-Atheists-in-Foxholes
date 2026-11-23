@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT=%SCRIPT_DIR%chunking.py"

where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python was not found on PATH.
    exit /b 1
)

python "%SCRIPT%" %*
exit /b %errorlevel%
