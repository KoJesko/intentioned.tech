@echo off
REM Intentioned MSI Builder
REM Requires WiX Toolset 3.11+ installed (https://wixtoolset.org/)

echo ========================================
echo  Intentioned MSI Installer Builder
echo ========================================
echo.

REM Check for WiX in PATH first
where candle >nul 2>&1
if %ERRORLEVEL% equ 0 goto :wix_in_path

REM Try WiX v3.14 path
set "CANDLE=C:\Program Files (x86)\WiX Toolset v3.14\bin\candle.exe"
set "LIGHT=C:\Program Files (x86)\WiX Toolset v3.14\bin\light.exe"
if exist "%CANDLE%" goto :wix_found

REM Try WiX v3.11 path
set "CANDLE=C:\Program Files (x86)\WiX Toolset v3.11\bin\candle.exe"
set "LIGHT=C:\Program Files (x86)\WiX Toolset v3.11\bin\light.exe"
if exist "%CANDLE%" goto :wix_found

echo ERROR: WiX Toolset not found!
echo Please install from: https://wixtoolset.org/releases/
echo Or install via winget: winget install WixToolset.WixToolset
pause
exit /b 1

:wix_in_path
set "CANDLE=candle"
set "LIGHT=light"
echo WiX found in PATH
goto :build_start

:wix_found
echo Found WiX at: %CANDLE%

:build_start

echo [1/4] Checking LICENSE.rtf...
if not exist LICENSE.rtf (
    echo ERROR: LICENSE.rtf not found! Please ensure it exists.
    pause
    exit /b 1
)
echo    Done.

echo [2/4] Compiling WiX source...
"%CANDLE%" installer.wxs -out installer.wixobj
if %ERRORLEVEL% neq 0 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)
echo    Done.

echo [3/4] Linking MSI package...
"%LIGHT%" installer.wixobj -ext WixUIExtension -out Intentioned-Setup.msi
if %ERRORLEVEL% neq 0 (
    echo ERROR: Linking failed!
    pause
    exit /b 1
)
echo    Done.

echo [4/4] Cleaning up...
del installer.wixobj 2>nul
echo    Done.

echo.
echo ========================================
echo  Build Complete!
echo  Output: Intentioned-Setup.msi
echo ========================================
echo.

dir Intentioned-Setup.msi

pause
