@echo off
chcp 65001 >nul
title Выделенный Сервер — Крымский Мост 1v1 (Порт 27015)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call run_windows.bat --self-test
)

echo Запуск выделенного сервера 1 на 1 на TCP-порту 27015...
.venv\Scripts\python main.py --server --port 27015
pause
