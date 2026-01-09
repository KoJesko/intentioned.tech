# Intentioned - Windows Installation Script
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

param(
    [switch]$NoConfigTool,
    [string]$InstallPath = "$env:LOCALAPPDATA\Intentioned"
)

$ErrorActionPreference = "Stop"

Write-Host @"
╔═══════════════════════════════════════════════════════════════╗
║         Intentioned - Social Skills Training Platform         ║
║                    Windows Installer v1.0                     ║
╚═══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check for Python
Write-Host "`n[1/7] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = $null
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "   Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
            Write-Host "   Please install Python 3.10 or later from https://python.org" -ForegroundColor Red
            exit 1
        }
        Write-Host "   Found: $pythonVersion ✓" -ForegroundColor Green
    }
} catch {
    Write-Host "   Python not found! Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Check for Git
Write-Host "`n[2/7] Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version 2>&1
    Write-Host "   Found: $gitVersion ✓" -ForegroundColor Green
} catch {
    Write-Host "   Git not found. Installing with winget..." -ForegroundColor Yellow
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
}

# Check for ffmpeg
Write-Host "`n[3/7] Checking ffmpeg installation..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "   Found: $ffmpegVersion ✓" -ForegroundColor Green
} catch {
    Write-Host "   ffmpeg not found. Installing with winget..." -ForegroundColor Yellow
    winget install --id BtbN.FFmpeg.GPL -e --source winget --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Create installation directory
Write-Host "`n[4/7] Creating installation directory..." -ForegroundColor Yellow
if (!(Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
}
Write-Host "   Path: $InstallPath ✓" -ForegroundColor Green

# Clone or update repository
Write-Host "`n[5/7] Downloading Intentioned..." -ForegroundColor Yellow
$repoPath = Join-Path $InstallPath "intentioned.tech"
if (Test-Path $repoPath) {
    Write-Host "   Updating existing installation..." -ForegroundColor Yellow
    Push-Location $repoPath
    git pull
    Pop-Location
} else {
    git clone https://github.com/KoJesko/intentioned.tech.git $repoPath
}
Write-Host "   Downloaded ✓" -ForegroundColor Green

# Create virtual environment and install dependencies
Write-Host "`n[6/7] Installing Python dependencies..." -ForegroundColor Yellow
Push-Location $repoPath
if (!(Test-Path "myenv")) {
    python -m venv myenv
}
& ".\myenv\Scripts\pip.exe" install --upgrade pip
& ".\myenv\Scripts\pip.exe" install -r requirements.txt
Pop-Location
Write-Host "   Dependencies installed ✓" -ForegroundColor Green

# Create start script and shortcuts
Write-Host "`n[7/7] Creating shortcuts..." -ForegroundColor Yellow

# Create start script
$startScript = @"
@echo off
cd /d "$repoPath"
call myenv\Scripts\activate
python server.py
pause
"@
$startScript | Out-File -FilePath (Join-Path $InstallPath "Start Intentioned.bat") -Encoding ASCII

# Create desktop shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Intentioned.lnk")
$Shortcut.TargetPath = Join-Path $InstallPath "Start Intentioned.bat"
$Shortcut.WorkingDirectory = $repoPath
$Shortcut.Description = "Intentioned - Social Skills Training"
$Shortcut.Save()

# Create config tool shortcut
$ConfigShortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Intentioned Config.lnk")
$ConfigShortcut.TargetPath = Join-Path $repoPath "myenv\Scripts\python.exe"
$ConfigShortcut.Arguments = "config_tool.py"
$ConfigShortcut.WorkingDirectory = $repoPath
$ConfigShortcut.Description = "Intentioned Configuration Tool"
$ConfigShortcut.Save()

Write-Host "   Shortcuts created on Desktop ✓" -ForegroundColor Green

# Add to Start Menu
$startMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Intentioned"
if (!(Test-Path $startMenuPath)) {
    New-Item -ItemType Directory -Path $startMenuPath -Force | Out-Null
}
Copy-Item "$env:USERPROFILE\Desktop\Intentioned.lnk" -Destination $startMenuPath
Copy-Item "$env:USERPROFILE\Desktop\Intentioned Config.lnk" -Destination $startMenuPath

Write-Host @"

╔═══════════════════════════════════════════════════════════════╗
║                  Installation Complete! 🎉                     ║
╠═══════════════════════════════════════════════════════════════╣
║  To start Intentioned:                                        ║
║    - Use the 'Intentioned' shortcut on your Desktop           ║
║    - Or run: Start Intentioned.bat                            ║
║                                                                ║
║  To configure:                                                 ║
║    - Use the 'Intentioned Config' shortcut on your Desktop    ║
║                                                                ║
║  Installation path: $InstallPath
╚═══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Green

# Offer to open config tool
if (!$NoConfigTool) {
    $openConfig = Read-Host "`nWould you like to configure Intentioned now? (Y/n)"
    if ($openConfig -ne "n" -and $openConfig -ne "N") {
        Write-Host "Opening Configuration Tool..." -ForegroundColor Cyan
        Push-Location $repoPath
        & ".\myenv\Scripts\python.exe" config_tool.py
        Pop-Location
    }
}

Write-Host "`nThank you for installing Intentioned!" -ForegroundColor Cyan
