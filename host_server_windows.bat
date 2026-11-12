@echo off
setlocal enableextensions
chcp 65001 >nul
title Crimea Bridge 1v1 - Dedicated Server (TCP 27015)
cd /d "%~dp0"

echo ==============================================================
echo   DEDICATED 1v1 SERVER - CRIMEA BRIDGE 1v1
echo   Players connect to this PC on TCP port 27015.
echo.
echo   The server automatically opens the port on your router (UPnP)
echo   and, if your provider hides the address (CGNAT), starts a FREE
echo   TCP tunnel. The ready-to-use address is printed below.
echo ==============================================================
echo.

call "%~dp0run_windows.bat" --server --port 27015
