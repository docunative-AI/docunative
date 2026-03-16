@echo off
REM DocuNative - llama.cpp Server Setup Script for Windows
REM Clones llama.cpp, compiles with CUDA/CPU fallback, starts server on port 8080
setlocal enabledelayedexpansion

set MODEL_TYPE=%~1
if "%MODEL_TYPE%"=="" set MODEL_TYPE=global

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%llama.cpp"
set "SERVER_PORT=8080"

REM Map model type to filename
if "%MODEL_TYPE%"=="global" (
    set "MODEL_FILE=weights/tiny-aya-global-q4_k_m.gguf"
) else if "%MODEL_TYPE%"=="earth" (
    set "MODEL_FILE=weights/tiny-aya-earth-q4_k_m.gguf"
) else (
    echo Error: Invalid model type '%MODEL_TYPE%'. Use 'global' or 'earth'.
    exit /b 1
)

set "MODEL_PATH=%SCRIPT_DIR%%MODEL_FILE%"

REM Check if model file exists
if not exist "%MODEL_PATH%" (
    echo Error: Model file not found: %MODEL_PATH%
    echo Please run: python models/pull_models.py
    exit /b 1
)

echo ==========================================
echo DocuNative - Inference Server Setup
echo ==========================================
echo Model: %MODEL_FILE%
echo Port: %SERVER_PORT%
echo.

REM Clone llama.cpp if not already present
if not exist "%REPO_DIR%" (
    echo [1/3] Cloning llama.cpp...
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "%REPO_DIR%"
    echo + Cloned llama.cpp
) else (
    echo [1/3] Using existing llama.cpp directory
)

cd /d "%REPO_DIR%"

REM Compile with appropriate backend
echo [2/3] Compiling llama.cpp...

REM Use vswhere to find any VS version (Community/Professional/Enterprise)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "tokens=*" %%i in ('"%VSWHERE%" -latest -property installationPath') do (
        call "%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
) else (
    echo Error: Visual Studio not found. Please install Visual Studio with C++ tools.
    echo Visit: https://visualstudio.microsoft.com/downloads/
    exit /b 1
)

REM Try CUDA build, fallback to CPU
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo   ^> Detected CUDA, building with CUDA support...
    call cmake -B build -DGGML_CUDA=1
) else (
    echo   ^> No CUDA detected, building for CPU...
    call cmake -B build
)

if errorlevel 1 (
    echo Error: CMake configuration failed.
    exit /b 1
)

call cmake --build build --config Release -j
if errorlevel 1 (
    echo Error: Build failed.
    exit /b 1
)

echo + Compilation complete

REM Find server binary (path differs by CMake version)
if exist ".\build\bin\Release\llama-server.exe" (
    set "SERVER_BIN=.\build\bin\Release\llama-server.exe"
) else if exist ".\build\bin\llama-server.exe" (
    set "SERVER_BIN=.\build\bin\llama-server.exe"
) else (
    echo Error: llama-server.exe not found after build.
    exit /b 1
)

REM Start the server
echo [3/3] Starting llama-server...
echo.
echo Server will be available at: http://localhost:%SERVER_PORT%
echo Health check: http://localhost:%SERVER_PORT%/health
echo Press Ctrl+C to stop the server
echo.

%SERVER_BIN% ^
    --model "%MODEL_PATH%" ^
    --port %SERVER_PORT% ^
    --host 0.0.0.0 ^
    --ctx-size 4096 ^
    --n-gpu-layers 99