@echo off
setlocal enableextensions
chcp 65001 >nul
title Crimea Bridge 1v1 - FPV Drone vs Fuel Truck (Desktop 3D Game)
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "LOGDIR=logs"

echo ==============================================================
echo   CRIMEA BRIDGE 1v1  -  FPV DRONE OPERATOR vs FUEL TRUCK
echo   Desktop 3D game. 1v1 multiplayer over TCP (port 27015).
echo ==============================================================
echo.

if exist "%PY%" goto run_game

echo [1/2] Creating virtual environment .venv ...
where python >nul 2>nul
if errorlevel 1 goto try_py_launcher
python -m venv .venv
if errorlevel 1 goto try_py_launcher
goto install_deps

:try_py_launcher
where py >nul 2>nul
if errorlevel 1 goto no_python
echo [1/2] Creating virtual environment .venv (via "py" launcher) ...
py -3 -m venv .venv
if errorlevel 1 goto venv_failed
goto install_deps

:no_python
echo.
echo [ERROR] Python 3.10+ was not found on this PC.
echo         1) Install Python from https://www.python.org/downloads/
echo         2) Tick the box "Add python.exe to PATH" during the setup
echo         3) Run this file again
goto failed

:venv_failed
echo.
echo [ERROR] Failed to create the virtual environment ".venv".
echo         Add the game folder to the antivirus exclusions and try again.
goto failed

:install_deps
echo [2/2] Installing game libraries (pygame-ce, numpy, pillow) ...
echo       This happens only once. Please keep the internet connection.
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto pip_failed
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed

:pip_failed
echo.
echo [ERROR] Installation of the libraries failed.
echo         Check the internet connection or proxy settings and run this file again.
echo         Make sure the archive was unpacked completely (do not run from inside the ZIP).
goto failed

:run_game
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>nul
if exist "%LOGDIR%\crash.log" del "%LOGDIR%\crash.log" >nul 2>nul
echo.
echo Starting the game ...
echo (the full log is written to logs\last_run.log)
echo.
"%PY%" main.py %*
set "RC=%ERRORLEVEL%"
echo.
echo ==============================================================
if "%RC%"=="0" goto ok_exit
echo   [ERROR] The game exited with code %RC%.
echo   Error report : logs\crash.log
echo   Full log     : logs\last_run.log
goto final
:ok_exit
echo   The game has been closed. Thank you for playing!
:final
echo ==============================================================
echo.
pause
exit /b %RC%

:failed
echo.
pause
exit /b 1
