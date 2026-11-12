"""
Тест SSH-туннеля (запасной путь) на эмуляторе ssh.

Подменяем системный ssh скриптом, который печатает публичный адрес
в формате Pinggy, и проверяем, что адрес извлекается верно.

Запуск:  python -m tools.test_ssh_tunnel
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.net_setup as net_setup  # noqa: E402
from src.net_setup import SshTunnel  # noqa: E402

FAKE_SSH = """#!/bin/sh
echo "Welcome to Pinggy!"
echo "Forwarding TCP traffic from tcp://rnsej-14-139-241-214.a.free.pinggy.link:42289"
sleep 30
"""


def _make_fake_ssh(directory: str) -> str:
    path = os.path.join(directory, "ssh")
    with open(path, "w", encoding="utf-8") as f:
        f.write(FAKE_SSH)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def run_test() -> int:
    problems = []

    tmp = tempfile.mkdtemp(prefix="sshfake_")
    _make_fake_ssh(tmp)
    os.environ["PATH"] = tmp + os.pathsep + os.environ.get("PATH", "")

    print(f"[1] ssh доступен: {SshTunnel.ssh_available()}")
    if not SshTunnel.ssh_available():
        problems.append("ssh не найден в PATH")

    tunnel = SshTunnel(local_host="127.0.0.1")
    ok = tunnel.start(27015, timeout=12.0)
    print(f"[2] Туннель поднят: {ok}, адрес={tunnel.public_address!r}, "
          f"ошибка={tunnel.error!r}, состояние={tunnel.state!r}")
    if not ok:
        problems.append("туннель не поднялся")
    elif tunnel.public_address != "rnsej-14-139-241-214.a.free.pinggy.link:42289":
        problems.append(f"адрес разобран неверно: {tunnel.public_address!r}")
    if not tunnel.running:
        problems.append("туннель не в состоянии running")

    tunnel.stop()
    print(f"[3] После stop: состояние={tunnel.state!r}, адрес={tunnel.public_address!r}")
    if tunnel.state != "stopped":
        problems.append("туннель не остановился")

    # --- Проверка: отсутствие ssh не должно ломать настройку ----------------
    os.environ["PATH"] = tempfile.mkdtemp(prefix="empty_")
    print(f"[4] ssh без PATH: {SshTunnel.ssh_available()}")
    t2 = SshTunnel()
    ok2 = t2.start(27015, timeout=5.0)
    print(f"[5] Результат без ssh: {ok2}, ошибка={t2.error!r}")
    if ok2:
        problems.append("туннель поднялся без ssh")
    if not t2.error:
        problems.append("нет понятной ошибки при отсутствии ssh")

    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print("[OK] SSH-туннель: адрес извлекается, остановка работает, отсутствие ssh обработано.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
