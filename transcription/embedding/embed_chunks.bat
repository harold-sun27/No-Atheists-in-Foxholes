@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "SCRIPT=%SCRIPT_DIR%embed_chunks.py"

if not exist "%PYTHON%" (
    echo Error: Embedding environment not found.
    echo Expected: %PYTHON%
    echo Follow the setup steps in the embedding README.
    exit /b 1
)

"%PYTHON%" "%SCRIPT%" %*
exit /b %errorlevel%
