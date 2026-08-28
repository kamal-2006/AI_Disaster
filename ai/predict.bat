@echo off
if "%~1"=="" (
    echo Usage: .\predict.bat ^<max_temp^> [humidity]
    echo Example: .\predict.bat 42.5 55
    exit /b 1
)

if "%~2"=="" (
    "%~dp0..\.venv\Scripts\python.exe" "%~dp0prediction\predict.py" --max-temp %~1
) else (
    "%~dp0..\.venv\Scripts\python.exe" "%~dp0prediction\predict.py" --max-temp %~1 --humidity %~2
)
