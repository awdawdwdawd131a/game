#!/usr/bin/env python3
"""
Крымский Мост 1v1 — Оператор FPV Дрона vs Водитель-Стрелок Бензовоза
Настольная 3D-игра для ПК (Windows / Linux / macOS) с сетевым мультиплеером 1 на 1.

Режимы запуска из командной строки:
  python main.py                        # Обычный запуск настольного 3D окна игры
  python main.py --verify               # Запуск окна сразу в Проверочном режиме (Автотесты + Сплит-скрин)
  python main.py --self-test            # Консольная автоматическая проверка работоспособности (без монитора / CI)
  python main.py --server --port 27015  # Выделенный headless-сервер 1v1 для хостинга на VPS или ПК
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def get_repo_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def run_headless_server(host: str, port: int, bot_fill: bool) -> int:
    from src.config import WEATHER_BY_ID
    from src.network import GameServer, get_local_lan_ip

    srv = GameServer(host=host, port=port, fill_with_bot=bot_fill)
    bound = srv.start()
    lan_ip = get_local_lan_ip()
    print("=" * 72)
    print(" ВЫДЕЛЕННЫЙ СЕРВЕР «КРЫМСКИЙ МОСТ 1v1» ЗАПУЩЕН!")
    print(f" Слушает адрес:  {host}:{bound}")
    print(f" Локальный/VPN IP этого ПК для подключения игроков: {lan_ip}:{bound}")
    print(f" Заполнение пустого слота ИИ-ботом: {'ВКЛ' if bot_fill else 'ВЫКЛ'}")
    print(" Нажмите Ctrl+C для остановки сервера.")
    print("=" * 72)

    stop_flag = False

    def _sig_handler(_sig, _frame):
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        while not stop_flag:
            d_conn, t_conn = srv.connected_roles()
            snap = srv.sim.get_snapshot()
            weather_ru = WEATHER_BY_ID.get(snap["weather_id"])
            weather_name = weather_ru.name_ru if weather_ru else snap["weather_id"]
            print(
                f"\r[Раунд {snap['round_number']} | {snap['phase']}] "
                f"Счёт FPV {snap['drone_score']}:{snap['truck_score']} Бензовоз | "
                f"Погода: {weather_name} | "
                f"Игроки: FPV={'ОНЛАЙН' if d_conn else '---'}, Бензовоз={'ОНЛАЙН' if t_conn else '---'}   ",
                end="",
                flush=True,
            )
            time.sleep(1.0)
    finally:
        print("\nОстановка сервера...")
        srv.stop()
    return 0


def run_cli_self_test(repo_root: str, screenshots_dir: str | None) -> int:
    # Ensure headless-safe SDL drivers if no display is attached
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from src.selftest import run_verification_suite

    print("=" * 76)
    print(" ЗАПУСК ПРОВЕРОЧНОГО РЕЖИМА («Крымский Мост 1v1 — FPV vs Бензовоз»)")
    print("=" * 76)

    def _printer(res) -> None:
        badge = "[ OK ]" if res.passed else "[FAIL]"
        print(f" {badge} {res.name_ru}")
        print(f"        └─ {res.details_ru} ({res.duration_ms:.1f} мс)")

    results = run_verification_suite(
        repo_root=repo_root,
        save_screenshots_dir=screenshots_dir,
        on_progress=_printer,
    )
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("-" * 76)
    if passed == total:
        print(f" ИТОГ: ВСЕ ПРОВЕРКИ УСПЕШНО ПРОЙДЕНЫ ({passed}/{total})! ИГРА ПОЛНОСТЬЮ РАБОТОСПОСОБНА.")
        return 0
    else:
        print(f" ИТОГ: ОБНАРУЖЕНЫ ОШИБКИ ({passed}/{total} пройдено).")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Настольная 3D-игра «Крымский Мост 1v1: FPV Дрон против Бензовоза»"
    )
    parser.add_argument("--server", action="store_true", help="Запустить выделенный сетевой сервер 1v1 (без 3D окна)")
    parser.add_argument("--host", default="0.0.0.0", help="IP-интерфейс сервера (по умолчанию 0.0.0.0)")
    parser.add_argument("--port", type=int, default=27015, help="TCP порт сервера (по умолчанию 27015)")
    parser.add_argument("--bot-fill", action="store_true", help="На сервере: подставлять ИИ-бота, пока подключён только 1 игрок")
    parser.add_argument("--verify", action="store_true", help="Открыть настольную игру сразу в Проверочном режиме (Сплит-Скрин + тесты)")
    parser.add_argument("--self-test", action="store_true", help="Выполнить полную автоматическую самопроверку в консоли и выйти")
    parser.add_argument("--save-test-frames", default=None, help="Папка для сохранения проверочных 3D-скриншотов при --self-test")
    parser.add_argument("--width", type=int, default=1280, help="Ширина окна игры")
    parser.add_argument("--height", type=int, default=720, help="Высота окна игры")

    args = parser.parse_args()
    repo_root = get_repo_root()

    if args.self_test:
        return run_cli_self_test(repo_root, args.save_test_frames)

    if args.server:
        return run_headless_server(args.host, args.port, args.bot_fill)

    from src.app import DesktopGameApp

    app = DesktopGameApp(
        repo_root=repo_root,
        width=args.width,
        height=args.height,
        start_mode="verify" if args.verify else "menu",
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
