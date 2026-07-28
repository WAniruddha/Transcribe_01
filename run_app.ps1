param(
    [string]$VenvPath = "D:\02_Applications\10_VEnv\E1"
)

$ErrorActionPreference = "Stop"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python was not found at $PythonExe. Check the -VenvPath argument."
}

# WinGet can install FFmpeg correctly without exposing its updated PATH to
# every new shell immediately. Find that installation and prepend its bin folder.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $WinGetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    $FFmpegExe = Get-ChildItem `
        $WinGetPackages `
        -Recurse `
        -Filter "ffmpeg.exe" `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*Gyan.FFmpeg.Essentials*" } |
        Select-Object -First 1

    if ($FFmpegExe) {
        $env:Path = "$($FFmpegExe.Directory.FullName);$env:Path"
    }
}

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

& $PythonExe -m streamlit run (Join-Path $PSScriptRoot "app.py")
