"""
Тест GUI: подключение к недоступному хосту НЕ должно замораживать окно.

Проверяет:
  1) _join_remote_match() возвращается мгновенно (окно не "лагает").
  2) Экран ожидания (JOINING) рисуется и показываетадрес-цели.
  3) ESC отменяет подключение.
  4) Ошибка подключения возвращает игрока на экран ввода адреса с понятным текстом.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from src.app import DesktopGameApp  # noqa: E402


def _frame(app: DesktopGameApp, dt: float = 1 / 60.0) -> None:
    app._update(dt)
    app._render()


def run_test() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = DesktopGameApp(repo_root=repo_root, width=960, height=600, start_mode="menu")

    problems = []

    # --- 1) Подключение к недоступному адресу -------------------------------
    app.state = "JOIN_SETUP"
    app.ip_input = "10.255.255.1"
    app.port_input = "27015"

    t0 = time.perf_counter()
    app._join_remote_match("10.255.255.1", 27015)
    elapsed = time.perf_counter() - t0
    print(f"[1] _join_remote_match вернулся за {elapsed:.3f} с (было 5+ с блокировки)")
    print(f"[2] Состояние экрана: {app.state!r}, цель: {app.join_target!r}")
    if elapsed > 0.5:
        problems.append(f"подключение блокировало GUI {elapsed:.2f} с")
    if app.state != "JOINING":
        problems.append(f"ожидался экран JOINING, а не {app.state!r}")

    # --- 2) Кадры идут, экран ожидания рисуется -----------------------------
    drawn = 0
    for _ in range(30):
        _frame(app)
        drawn += 1
        time.sleep(0.005)
    print(f"[3] Отрисовано {drawn} кадров на экране ожидания, состояние={app.state!r}")
    if app.state != "JOINING":
        problems.append(f"состояние изменилось раньше времени: {app.state!r}")

    # --- 3) ESC отменяет подключение ----------------------------------------
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode="\x1b"))
    app._process_events()
    print(f"[4] После ESC: состояние={app.state!r}, баннер={app.status_banner!r}")
    if app.state != "JOIN_SETUP":
        problems.append(f"ESC не вернул на экран ввода: {app.state!r}")
    if app.net_client is not None:
        problems.append("клиент не был очищен после отмены")

    # --- 4) Ошибка подключения показывается текстом -------------------------
    app.state = "JOIN_SETUP"
    app.ip_input = "127.0.0.1"
    app.port_input = "1"          # порт, который никто не слушает
    app._join_remote_match("127.0.0.1", 1)
    t1 = time.perf_counter()
    deadline = t1 + 8.0
    while app.state == "JOINING" and time.perf_counter() < deadline:
        _frame(app)
        time.sleep(0.02)
    print(f"[5] После неудачного подключения: состояние={app.state!r}, "
          f"баннер={app.status_banner!r}")
    if app.state != "JOIN_SETUP":
        problems.append(f"после ошибки ожидался JOIN_SETUP, а не {app.state!r}")
    if not app.status_banner:
        problems.append("нет текста ошибки для пользователя")

    app.running = False
    pygame.quit()

    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print("[OK] GUI не зависает при подключении: есть экран ожидания, отмена и понятные ошибки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
