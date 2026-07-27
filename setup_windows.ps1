param(
    [string]$VenvPath = "D:\02_Applications\10_VEnv\E1"
)

$ErrorActionPreference = "Stop"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python was not found at $PythonExe. Create or correct the virtual environment first."
}

Write-Host "Using Python: $PythonExe"
& $PythonExe -m pip install --upgrade pip

& $PythonExe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyTorch is not installed in E1. Install the correct CPU or NVIDIA build first; see README.md."
    exit 1
}

& $PythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
& $PythonExe -m py_compile (Join-Path $PSScriptRoot "app.py")

Write-Host ""
Write-Host "Setup complete. Start the app with:"
Write-Host ".\run_app.ps1"
