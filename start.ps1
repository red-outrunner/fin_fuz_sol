# Powershell Startup Script for Windows (Equivalent to start.sh)
# Starts Backend (FastAPI via uvicorn) and Frontend (Vite)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting Financial Application Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Determine Root Directory
$RootDir = $PSScriptRoot
if (-not $RootDir) { $RootDir = Get-Location }

# Backend check
$BackendDir = Join-Path $RootDir "backend"
if (-not (Test-Path $BackendDir)) {
    Write-Host "Error: backend directory not found at $BackendDir" -ForegroundColor Red
    exit 1
}

# Locate Python executable in backend venv or system
$PythonPath = $null
$PossiblePythonPaths = @(
    (Join-Path $BackendDir "venv\Scripts\python.exe"),
    (Join-Path $BackendDir ".venv\Scripts\python.exe"),
    (Join-Path $BackendDir "Scripts\python.exe")
)

foreach ($path in $PossiblePythonPaths) {
    if (Test-Path $path) {
        $PythonPath = $path
        break
    }
}

if (-not $PythonPath) {
    $sysPy = Get-Command python -ErrorAction SilentlyContinue
    if ($sysPy) {
        $PythonPath = $sysPy.Source
    } else {
        Write-Host "ERROR: No Python virtual environment or system Python found in backend." -ForegroundColor Red
        Write-Host "Create it with: cd backend; python -m venv venv; .\venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Using Python: $PythonPath" -ForegroundColor Green

# Check uvicorn
$uvicornCheck = & $PythonPath -c "import uvicorn; print('ok')" 2>$null
if ($uvicornCheck -ne "ok") {
    Write-Host "ERROR: uvicorn is not installed in $PythonPath." -ForegroundColor Red
    Write-Host "Install dependencies with: cd backend; $PythonPath -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Frontend check
$FrontendDir = Join-Path $RootDir "frontend"
if (-not (Test-Path $FrontendDir)) {
    Write-Host "Error: frontend directory not found at $FrontendDir" -ForegroundColor Red
    exit 1
}

# Start Backend
Write-Host "`nStarting Backend (uvicorn on http://127.0.0.1:8000)..." -ForegroundColor Cyan
$BackendProcess = Start-Process -FilePath $PythonPath -ArgumentList "-m uvicorn main:app --reload" -WorkingDirectory $BackendDir -PassThru

# Start Frontend
Write-Host "Starting Frontend (Vite dev server on http://localhost:3000)..." -ForegroundColor Cyan
$npmCmd = if (Get-Command "npm.cmd" -ErrorAction SilentlyContinue) { "npm.cmd" } else { "npm" }
$FrontendProcess = Start-Process -FilePath $npmCmd -ArgumentList "run dev" -WorkingDirectory $FrontendDir -PassThru

Write-Host "`n[SUCCESS] Both services started successfully!" -ForegroundColor Green
Write-Host "  - Backend PID:  $($BackendProcess.Id) (http://127.0.0.1:8000)" -ForegroundColor Gray
Write-Host "  - Frontend PID: $($FrontendProcess.Id) (http://localhost:3000)" -ForegroundColor Gray
Write-Host "`nPress Ctrl+C to stop all services..." -ForegroundColor Yellow

try {
    while (-not $BackendProcess.HasExited -and -not $FrontendProcess.HasExited) {
        Start-Sleep -Milliseconds 500
    }
}
catch {
    # Gracefully catch interrupt / Ctrl+C
}
finally {
    Write-Host "`nStopping services..." -ForegroundColor Yellow
    if ($BackendProcess -and -not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($FrontendProcess -and -not $FrontendProcess.HasExited) {
        Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "All services stopped cleanly." -ForegroundColor Green
}
