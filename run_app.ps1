param(
    [string]$VenvPath = "D:\02_Applications\10_VEnv\E1"
)

$ErrorActionPreference = "Stop"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python was not found at $PythonExe. Check the -VenvPath argument."
}

& $PythonExe -m streamlit run (Join-Path $PSScriptRoot "app.py")
