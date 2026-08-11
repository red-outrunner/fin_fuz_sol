@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  Starting Financial Application Services
echo ========================================

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"
set "FRONTEND_DIR=%ROOT_DIR%frontend"

if not exist "%BACKEND_DIR%" (
    echo Error: backend directory not found at %BACKEND_DIR%
    pause
    exit /b 1
)

if not exist "%FRONTEND_DIR%" (
    echo Error: frontend directory not found at %FRONTEND_DIR%
    pause
    exit /b 1
)

set "PY="
if exist "%BACKEND_DIR%\venv\Scripts\python.exe" (
    set "PY=%BACKEND_DIR%\venv\Scripts\python.exe"
) else if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
    set "PY=%BACKEND_DIR%\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Using Python: %PY%

"%PY%" -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo ERROR: uvicorn is not installed in %PY%
    echo Please run: cd backend ^& "%PY%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Backend (FastAPI on http://127.0.0.1:8000)...
start "Backend - FastAPI" /D "%BACKEND_DIR%" "%PY%" -m uvicorn main:app --reload

echo Starting Frontend (Vite on http://localhost:3000)...
start "Frontend - Vite" /D "%FRONTEND_DIR%" cmd /c npm run dev

echo ========================================
echo Both Backend and Frontend services are running in separate windows.
echo Close those windows to stop the services.
echo ========================================
pause
