@echo off
chcp 65001 >nul
title Сборка EXE — Крымский Мост 1v1
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)

echo Сборка автономного настольного файла CrimeanBridge1v1.exe...
.venv\Scripts\pyinstaller --noconfirm --onefile --windowed --name "CrimeanBridge1v1" main.py
copy /Y "черешня.jpg" "dist\черешня.jpg" >nul

echo ======================================================================
echo Готово! Исполняемый файл для ПК создан в папке: dist\CrimeanBridge1v1.exe
echo Вы можете скопировать папку dist на второй компьютер и запускать без Python!
echo ======================================================================
pause
