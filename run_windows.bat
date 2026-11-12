@echo off
chcp 65001 >nul
title Крымский Мост 1v1 — FPV vs Бензовоз (Desktop 3D)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Первичная настройка виртуального окружения Python...
    python -m venv .venv
    if errorlevel 1 (
        echo ОШИБКА: Python 3.10+ не найден! Установите Python с https://www.python.org/downloads/ и отметьте галочку "Add Python to PATH".
        pause
        exit /b 1
    )
    echo [2/2] Установка библиотек игры...
    .venv\Scripts\pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt
)

echo Запуск настольной 3D-игры «Крымский Мост 1v1»...
.venv\Scripts\python main.py %*
