@echo off
setlocal enableextensions
chcp 65001 >nul
title Build EXE - Crimea Bridge 1v1
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if exist "%PY%" goto have_venv
echo [1/3] Creating virtual environment .venv ...
where python >nul 2>nul
if errorlevel 1 goto no_python
python -m venv .venv
if errorlevel 1 goto venv_failed
goto install_deps

:have_venv
echo [1/3] Virtual environment found.
goto install_deps

:no_python
echo.
echo [ERROR] Python 3.10+ was not found. Install it from https://www.python.org/downloads/
pause
exit /b 1

:venv_failed
echo.
echo [ERROR] Failed to create the virtual environment ".venv".
pause
exit /b 1

:install_deps
echo [2/3] Installing build dependencies (pyinstaller) ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto pip_failed
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed

echo [3/3] Building the standalone executable ...
if /i "%~1"=="debug" goto build_debug
"%PY%" -m PyInstaller --noconfirm --onefile --windowed --name "CrimeanBridge1v1" main.py
if errorlevel 1 goto build_failed
copy /Y *.jpg "dist\" >nul 2>nul
goto build_done

:build_debug
echo       (debug build - the console window stays visible)
"%PY%" -m PyInstaller --noconfirm --onefile --console --name "CrimeanBridge1v1-debug" main.py
if errorlevel 1 goto build_failed
copy /Y *.jpg "dist\" >nul 2>nul
goto build_done

:pip_failed
echo.
echo [ERROR] Installation of the libraries failed. Check the internet connection.
pause
exit /b 1

:build_failed
echo.
echo [ERROR] PyInstaller build failed. See the messages above.
pause
exit /b 1

:build_done
echo.
echo ==============================================================
echo   DONE! The standalone game is in the "dist" folder:
echo     dist\CrimeanBridge1v1.exe
echo   Copy the whole "dist" folder to your friend's PC -
echo   it runs without Python installed.
echo   A crash report is written to logs\crash.log next to the .exe
echo ==============================================================
echo.
pause
