@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "SCRIPT=%SCRIPT_DIR%transcribe.py"

if not exist "%PYTHON%" (
    echo Error: Project Python environment not found.
    echo Expected: %PYTHON%
    echo Run the project setup first.
    exit /b 1
)

"%PYTHON%" "%SCRIPT%" %*
exit /b %errorlevel%
