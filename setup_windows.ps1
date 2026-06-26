# setup_windows.ps1
# ==========================================
# Full Windows setup for llm-reliability-monitor
# Run this once in PowerShell (as Administrator) before any experiments.
#
# Prerequisites this script assumes:
#   - Windows 10/11 64-bit
#   - NVIDIA GPU with CUDA-capable drivers installed
#     (check: nvidia-smi should work in a terminal)
#   - Internet access
#
# Usage:
#   Right-click PowerShell -> "Run as Administrator"
#   cd to the folder where this script lives, then:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#     .\setup_windows.ps1
# ==========================================

$ErrorActionPreference = "Stop"

function Log($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Warn($msg){ Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Fail($msg){ Write-Host "    [FAIL] $msg" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------
# 0. Check we are running as Administrator
# ------------------------------------------------------------------
Log "Checking administrator privileges..."
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Fail "Please re-run PowerShell as Administrator." }
Ok "Running as Administrator"

# ------------------------------------------------------------------
# 1. Check nvidia-smi (GPU driver)
# ------------------------------------------------------------------
Log "Checking NVIDIA GPU driver..."
try {
    $smi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Ok "GPU detected: $smi"
} catch {
    Warn "nvidia-smi not found. Make sure NVIDIA drivers are installed."
    Warn "Download from: https://www.nvidia.com/drivers"
    Warn "Continuing anyway — smoke_test.py will still work without a GPU."
}

# ------------------------------------------------------------------
# 2. Install Python 3.11 via winget if not present
# ------------------------------------------------------------------
Log "Checking Python..."
$pythonOk = $false
try {
    $pyVer = python --version 2>&1
    if ($pyVer -match "3\.(10|11|12)") {
        Ok "Python found: $pyVer"
        $pythonOk = $true
    } else {
        Warn "Found $pyVer but need 3.10+. Will install 3.11."
    }
} catch {
    Warn "Python not found. Will install."
}

if (-not $pythonOk) {
    Log "Installing Python 3.11 via winget..."
    winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $pyVer = python --version 2>&1
    Ok "Installed: $pyVer"
}

# ------------------------------------------------------------------
# 3. Upgrade pip
# ------------------------------------------------------------------
Log "Upgrading pip..."
python -m pip install --upgrade pip --quiet
Ok "pip upgraded"

# ------------------------------------------------------------------
# 4. Install CUDA-enabled PyTorch (CUDA 12.1 wheel)
#    Adjust the index URL if your CUDA version differs.
#    Check your CUDA version with: nvidia-smi (top-right corner shows CUDA version)
#    CUDA 11.8: https://download.pytorch.org/whl/cu118
#    CUDA 12.1: https://download.pytorch.org/whl/cu121
#    CUDA 12.4: https://download.pytorch.org/whl/cu124
# ------------------------------------------------------------------
Log "Installing PyTorch with CUDA 12.1 support..."
Warn "If your CUDA version is different, edit the --index-url in this script."
Warn "Check your CUDA version: run 'nvidia-smi' and look at the top-right corner."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
Ok "PyTorch installed"

# ------------------------------------------------------------------
# 5. Install project requirements
# ------------------------------------------------------------------
Log "Installing project requirements from requirements.txt..."
if (-not (Test-Path "requirements.txt")) {
    Fail "requirements.txt not found. Run this script from the llm-reliability-monitor directory."
}
python -m pip install -r requirements.txt --quiet
Ok "requirements.txt installed"

# ------------------------------------------------------------------
# 6. Install HuggingFace CLI
# ------------------------------------------------------------------
Log "Installing HuggingFace Hub CLI..."
python -m pip install "huggingface_hub[cli]" --quiet
Ok "huggingface_hub installed"

# ------------------------------------------------------------------
# 7. Install PyYAML (needed by run_experiment.py for thresholds.yaml)
# ------------------------------------------------------------------
Log "Ensuring PyYAML is installed..."
python -m pip install pyyaml --quiet
Ok "PyYAML installed"

# ------------------------------------------------------------------
# 8. Verify PyTorch sees the GPU
# ------------------------------------------------------------------
Log "Verifying PyTorch GPU access..."
$gpuCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print('  GPU', i, ':', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]" 2>&1
Write-Host $gpuCheck
if ($gpuCheck -match "CUDA available: True") {
    Ok "PyTorch can see your GPU"
} else {
    Warn "PyTorch reports no CUDA GPU. Experiments requiring a model will fail."
    Warn "Make sure your CUDA version matches the PyTorch wheel you installed."
    Warn "Smoke test (no GPU) will still work."
}

# ------------------------------------------------------------------
# 9. Run smoke test (no GPU required)
# ------------------------------------------------------------------
Log "Running smoke_test.py (no GPU required)..."
python scripts/smoke_test.py
if ($LASTEXITCODE -ne 0) {
    Fail "Smoke test failed. Fix the issues above before continuing."
}

# ------------------------------------------------------------------
# 10. HuggingFace login
# ------------------------------------------------------------------
Log "HuggingFace login..."
Write-Host ""
Write-Host "    You need a HuggingFace account and a token to download Llama-3-8B." -ForegroundColor Yellow
Write-Host "    1. Go to: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
Write-Host "    2. Create a token with 'Read' access" -ForegroundColor Yellow
Write-Host "    3. Accept the Llama-3 license at:" -ForegroundColor Yellow
Write-Host "       https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct" -ForegroundColor Yellow
Write-Host ""
huggingface-cli login
Ok "HuggingFace login complete"

# ------------------------------------------------------------------
# 11. Download model checkpoints
# ------------------------------------------------------------------
Log "Downloading model checkpoints (this will take a while and ~30GB of disk)..."
Write-Host "    Downloading: all-MiniLM-L6-v2, meta-llama/Meta-Llama-3-8B-Instruct" -ForegroundColor Yellow
Write-Host "    You can Ctrl+C and re-run 'python scripts/download_models.py' separately" -ForegroundColor Yellow
Write-Host "    if you want to do this step later." -ForegroundColor Yellow
Write-Host ""
python scripts/download_models.py

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Generate tasks:"
Write-Host "     python scripts/generate_tasks.py --n-tasks 1000 --seed 0 --out data/tasks.json"
Write-Host ""
Write-Host "  2. Run reduced experiment (F1 + F3, 3 seeds, ~16 GPU-hrs):"
Write-Host "     python scripts/run_experiment.py ``"
Write-Host "         --model meta-llama/Meta-Llama-3-8B-Instruct ``"
Write-Host "         --tasks data/tasks.json ``"
Write-Host "         --scenarios F1 F3 ``"
Write-Host "         --seeds 42 43 44 ``"
Write-Host "         --out results/llama3_8b_reduced.csv"
Write-Host ""
Write-Host "  3. Or run the full benchmark (all 6 scenarios, 5 seeds, ~48-72 GPU-hrs):"
Write-Host "     See README.md for the full command sequence."
Write-Host ""
