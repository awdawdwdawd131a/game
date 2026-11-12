#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/2] Создание виртуального окружения .venv..."
    python3 -m venv .venv
    echo "[2/2] Установка зависимостей..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/python main.py "$@"
