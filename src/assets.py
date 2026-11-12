"""
Поиск ресурсов (звуки, картинки), который работает и в собранном .exe.

В PyInstaller-режиме (--onefile) файлы распаковываются во временную папку
`sys._MEIPASS`, а рядом с .exe лежат только те файлы, которые мы туда
скопировали вручную. Поэтому ищем ресурс сразу во всех возможных местах.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


def bundle_root() -> str:
    """Временная папка PyInstaller (пустая, если запущено из исходников)."""
    return getattr(sys, "_MEIPASS", "") or ""


def candidate_paths(rel_path: str, repo_root: str = "") -> List[str]:
    """Все возможные пути к ресурсу - от самого вероятного к запасным."""
    rel_path = rel_path.replace("\\", "/").replace("/", os.sep)
    paths: List[str] = []

    if repo_root:
        paths.append(os.path.join(repo_root, rel_path))

    meipass = bundle_root()
    if meipass:
        paths.append(os.path.join(meipass, rel_path))

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths.append(os.path.join(here, rel_path))

    exe_dir = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else ""
    if exe_dir:
        paths.append(os.path.join(exe_dir, rel_path))

    paths.append(os.path.join(os.getcwd(), rel_path))
    return paths


def find_file(rel_path: str, repo_root: str = "") -> Optional[str]:
    """Возвращает путь к существующему файлу или None."""
    for candidate in candidate_paths(rel_path, repo_root):
        if os.path.isfile(candidate):
            return candidate
    return None


def find_dir(rel_path: str, repo_root: str = "") -> Optional[str]:
    """Возвращает путь к существующей папке или None."""
    for candidate in candidate_paths(rel_path, repo_root):
        if os.path.isdir(candidate):
            return candidate
    return None
