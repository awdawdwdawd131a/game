@echo off
setlocal enableextensions
chcp 65001 >nul
title Build EXE - Crimea Bridge 1v1
cd /d "%~dp0"

REM ===========================================================================
REM  Сборка полностью автономных .exe файлов (без Python у получателя).
REM
REM  Итог:
REM    dist\CrimeanBridge1v1.exe        - игра для обоих игроков (друг)
REM    dist\CrimeanBridge1v1-Server.exe - выделенный сервер для хоста
REM
REM  Все ресурсы (звуки, картинки) встраиваются ВНУТРЬ .exe.
REM ===========================================================================

set "PY=.venv\Scripts\python.exe"

if exist "%PY%" goto have_venv
echo [1/4] Creating virtual environment .venv ...
where python >nul 2>nul
if errorlevel 1 goto no_python
python -m venv .venv
if errorlevel 1 goto venv_failed
goto install_deps

:have_venv
echo [1/4] Virtual environment found.
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
echo [2/4] Installing build dependencies ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto pip_failed
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed

echo [3/4] Building the standalone executables ...

REM  --add-data "assets;assets"  -> все звуки и картинки ВНУТРИ exe
REM  --noupx                   -> не сжимать (иначе antivirus ругается)
REM  --console / --windowed    -> для сервера нужен видимый вывод адресов

"%PY%" -m PyInstaller --noconfirm --clean --noupx --onefile --windowed ^
    --name "CrimeanBridge1v1" ^
    --add-data "assets;assets" ^
    --add-data "черешня.jpg;." ^
    --hidden-import "src.net_setup" ^
    --hidden-import "src.assets" ^
    main.py
if errorlevel 1 goto build_failed

"%PY%" -m PyInstaller --noconfirm --clean --noupx --onefile --console ^
    --name "CrimeanBridge1v1-Server" ^
    --add-data "assets;assets" ^
    --add-data "черешня.jpg;." ^
    --hidden-import "src.net_setup" ^
    --hidden-import "src.assets" ^
    main.py
if errorlevel 1 goto build_failed

echo [4/4] Checking the result ...
if not exist "dist\CrimeanBridge1v1.exe" goto build_failed
if not exist "dist\CrimeanBridge1v1-Server.exe" goto build_failed

REM  Справочные файлы (не обязательны для работы .exe)
copy /Y "черешня.jpg" "dist\" >nul 2>nul
copy /Y "HOSTING_RU.md" "dist\" >nul 2>nul
copy /Y "README.md" "dist\" >nul 2>nul

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
echo   DONE! Standalone builds are in the "dist" folder:
echo.
echo     dist\CrimeanBridge1v1.exe          - the game (for your friend)
echo     dist\CrimeanBridge1v1-Server.exe   - the host server
echo
echo   Each .exe is a SINGLE self-contained file:
echo   no Python, no project folders, no extra files needed.
echo
echo   Send the friend ONLY:  CrimeanBridge1v1.exe
echo   Keep for yourself:      CrimeanBridge1v1-Server.exe
echo
echo   Crash reports (if any) are written to logs\crash.log
echo   next to the .exe file.
echo ==============================================================
echo.
pause
