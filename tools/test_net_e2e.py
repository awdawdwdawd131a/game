"""
Сквозной тест: игра-сервер + бесплатный туннель bore + клиент через туннель.

Проверяет, что клиент может подключиться к игровому серверу ТОЛЬКО ЧЕРЕЗ
туннель (как это будет у пользователя и его друга из разных сетей).
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.net_setup import BoreTunnel  # noqa: E402
from src.network import GameClient, GameServer  # noqa: E402
from tools.test_bore_tunnel import MockBoreServer  # noqa: E402


def run_test() -> int:
    srv = GameServer(host="127.0.0.1", port=0, fill_with_bot=False)
    bound = srv.start()
    print(f"[1] Игровой сервер слушает 127.0.0.1:{bound}")

    mock = MockBoreServer()
    mock.start()
    print(f"[2] Туннель-сервер: control={mock.port}, публичный={mock.public_port}")

    tunnel = BoreTunnel(server="127.0.0.1", control_port=mock.port, local_host="127.0.0.1")
    ok = tunnel.start(bound)
    print(f"[3] Туннель поднят: {ok} -> {tunnel.public_address!r} ошибка={tunnel.error!r}")
    if not ok:
        mock.stop()
        srv.stop()
        return 1

    # Клиент подключается через публичный адрес туннеля
    host, _, port_s = tunnel.public_address.rpartition(":")
    cli = GameClient()
    t0 = time.perf_counter()
    cli.connect_async(host, int(port_s), preferred_role="drone", timeout=8.0)
    res = None
    while res is None:
        res = cli.poll_connect()
        time.sleep(0.02)
    print(f"[4] Клиент через туннель: результат={res}, роль={cli.role!r}, "
          f"за {time.perf_counter() - t0:.2f}с, ошибка={cli.error_msg!r}")

    problems = []
    if res is not True:
        problems.append(f"клиент не подключился: {cli.error_msg}")
    else:
        # Проверяем, что сервер реально видит игрока
        time.sleep(0.5)
        d_conn, t_conn = srv.connected_roles()
        print(f"[5] Сервер видит: drone={d_conn}, truck={t_conn}")
        if not d_conn and not t_conn:
            problems.append("сервер не видит подключившегося игрока")

        # Туннель не должен рваться при простое (регрессия таймаута сокета)
        time.sleep(2.5)
        d_conn2, t_conn2 = srv.connected_roles()
        cli.send_ping()
        time.sleep(0.4)
        print(f"[6] После простоя 2.5с: drone={d_conn2}, truck={t_conn2}, "
              f"пинг={cli.ping_ms:.0f} мс, connected={cli.connected}")
        if not (d_conn2 or t_conn2):
            problems.append("туннель оборвался при простое")
        if not cli.connected:
            problems.append("клиент отключился при простое")

    cli.disconnect()
    tunnel.stop()
    mock.stop()
    srv.stop()

    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print("[OK] Сквозной тест через туннель пройден: сервер + туннель + клиент.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
