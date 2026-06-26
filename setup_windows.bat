@echo off
REM setup_windows.bat
REM ==========================================
REM Lightweight CMD alternative to setup_windows.ps1
REM Use this if you cannot run PowerShell scripts.
REM Run from the llm-reliability-monitor directory.
REM ==========================================

echo.
echo === LLM Reliability Monitor - Windows Setup ===
echo.

REM Check Python
echo [1/7] Checking Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found.
    echo Download Python 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
python --version
echo.

REM Upgrade pip
echo [2/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Done.
echo.

REM Install PyTorch (CUDA 12.1)
echo [3/7] Installing PyTorch (CUDA 12.1)...
echo If your CUDA version differs, edit this file and change the --index-url.
echo Check CUDA version: run nvidia-smi and look top-right.
echo   CUDA 11.8: https://download.pytorch.org/whl/cu118
echo   CUDA 12.1: https://download.pytorch.org/whl/cu121
echo   CUDA 12.4: https://download.pytorch.org/whl/cu124
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo Done.
echo.

REM Install requirements
echo [4/7] Installing requirements.txt...
if not exist requirements.txt (
    echo ERROR: requirements.txt not found.
    echo Run this script from the llm-reliability-monitor directory.
    pause
    exit /b 1
)
python -m pip install -r requirements.txt --quiet
python -m pip install "huggingface_hub[cli]" pyyaml --quiet
echo Done.
echo.

REM Check GPU
echo [5/7] Checking GPU visibility in PyTorch...
python -c "import torch; print('CUDA:', torch.cuda.is_available()); [print(' GPU', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
echo.

REM Smoke test
echo [6/7] Running smoke_test.py (no GPU needed)...
python scripts/smoke_test.py
if %ERRORLEVEL% neq 0 (
    echo FAILED: Smoke test did not pass. Fix errors above before continuing.
    pause
    exit /b 1
)
echo.

REM HuggingFace login
echo [7/7] HuggingFace login...
echo.
echo  You need a HuggingFace token to download Llama-3-8B:
echo  1. Go to: https://huggingface.co/settings/tokens
echo  2. Create a token with Read access
echo  3. Accept the license at:
echo     https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
echo.
huggingface-cli login
echo.

echo ==========================================
echo  Setup complete!
echo ==========================================
echo.
echo Next - download models (30GB, takes a while):
echo   python scripts/download_models.py
echo.
echo Then generate tasks:
echo   python scripts/generate_tasks.py --n-tasks 1000 --seed 0 --out data/tasks.json
echo.
echo Then run the reduced experiment (F1+F3, 3 seeds, ~16 GPU-hrs):
echo   python scripts/run_experiment.py ^
echo       --model meta-llama/Meta-Llama-3-8B-Instruct ^
echo       --tasks data/tasks.json ^
echo       --scenarios F1 F3 ^
echo       --seeds 42 43 44 ^
echo       --out results/llama3_8b_reduced.csv
echo.
pause
