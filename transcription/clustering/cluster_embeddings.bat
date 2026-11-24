@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "SCRIPT=%SCRIPT_DIR%cluster_embeddings.py"

if not exist "%PYTHON%" (
    echo Error: Clustering environment not found.
    echo Expected: %PYTHON%
    echo Follow the setup steps in the clustering README.
    exit /b 1
)

"%PYTHON%" "%SCRIPT%" %*
exit /b %errorlevel%
