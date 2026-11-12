#!/usr/bin/env python3
"""
Krymskiy Most 1v1 - Operator FPV droona vs Voditel-strelok benzovaza
Nastolnaya 3D-igra dlya PK (Windows / Linux / macOS) s setevym multilevelerom 1 na 1.

Rezhimy zapuska iz komandnoy stroki:
  python main.py                        # Obychnyy zapusk nastolnogo 3D okna igry
  python main.py --verify               # Zapusk okna srazu v Proverochnom rezhime (Avtotesty + Split-ekran)
  python main.py --self-test            # Konsolnaya avtomaticheskaya proverka (bez monitora / CI)
  python main.py --server --port 27015  # Vydelennyy headless-server 1v1 dlya hostinga na VPS ili PK

Vse soobshcheniya ob oshibkah pishutsya v logs/crash.log i v logs/last_run.log,
choby okno konsoli nikogda ne zakryvalos "molcha".
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback


# -----------------------------------------------------------------------------
# Robust console / logging helpers (Windows console & PyInstaller --windowed safe)
# -----------------------------------------------------------------------------
class _TeeWriter:
    """Writes to the original stream AND to a log file (tee without external tools)."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log = log_file

    @property
    def buffer(self):
        return getattr(self._stream, "buffer", None)

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    def writable(self) -> bool:
        return True


class _NullWriter:
    """Used when the process was started without a console (PyInstaller --windowed)."""

    def write(self, data: str) -> int:
        return len(data)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("no fileno")

    def writable(self) -> bool:
        return False


class _LogOnlyWriter:
    """
    Записывает вывод только в лог-файл (когда консоли нет, но лог нужен).

    В режиме --windowed у exe нет ни консоли, ни stdout, поэтому без этого
    класса игра запускается «вслепую» и у пользователя нет никакого следа
    о том, что произошло. Файл logs\last_run.log рядом с .exe решает эту
    задачу: по нему видно и нормальный запуск, и любую ошибку.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, log_file) -> None:
        self._log = log_file

    def write(self, data: str) -> int:
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("no fileno")

    def writable(self) -> bool:
        return True


def get_repo_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _log_dir(repo_root: str) -> str:
    d = os.path.join(repo_root, "logs")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _safe_print(*args, **kwargs) -> None:
    """print() that never raises (broken pipe, closed stdout, cp866 console, ...)."""
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            msg = " ".join(str(a) for a in args)
            enc = msg.encode("ascii", "replace").decode("ascii")
            sys.stdout.write(enc + "\n")
        except Exception:
            pass


def _prepare_streams(repo_root: str) -> None:
    """Re-encode console to UTF-8 (Russian Windows uses cp866) and tee output to a log file."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            setattr(sys, name, _NullWriter())
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    try:
        log_path = os.path.join(_log_dir(repo_root), "last_run.log")
        log_file = open(log_path, "a", encoding="utf-8", errors="replace")
        if sys.stdout is not None and not isinstance(sys.stdout, _NullWriter):
            sys.stdout = _TeeWriter(sys.stdout, log_file)
        else:
            # Консоли нет (--windowed) - пишем только в лог-файл рядом с .exe
            sys.stdout = _LogOnlyWriter(log_file)
        if sys.stderr is not None and not isinstance(sys.stderr, _NullWriter):
            sys.stderr = _TeeWriter(sys.stderr, log_file)
        else:
            sys.stderr = _LogOnlyWriter(log_file)
    except Exception:
        pass


def _write_crash_log(repo_root: str, text: str) -> str:
    path = os.path.join(_log_dir(repo_root), "crash.log")
    try:
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"КРИТИЧЕСКАЯ ОШИБКА — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 78 + "\n")
            f.write(text)
            f.write("\n")
    except Exception:
        pass
    return path


def _show_error_box(title: str, message: str) -> None:
    """Native Windows message box (used by the --windowed .exe build)."""
    try:
        if sys.platform.startswith("win"):
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)  # MB_ICONERROR
    except Exception:
        pass


def _install_crash_handler(repo_root: str) -> None:
    def handler(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log_path = _write_crash_log(repo_root, text)
        _safe_print("")
        _safe_print("=" * 78)
        _safe_print("КРИТИЧЕСКАЯ ОШИБКА ИГРЫ")
        _safe_print("=" * 78)
        _safe_print(text)
        _safe_print(f"Полный отчёт сохранён в файле: {log_path}")
        _safe_print("=" * 78)
        _safe_print("")
        _show_error_box(
            "Крымский Мост 1v1 — ошибка",
            f"Игра завершилась с ошибкой.\n\n"
            f"Подробности сохранены в файле:\n{log_path}\n\n"
            f"{exc_type.__name__}: {exc}",
        )

    sys.excepthook = handler


def _print_banner() -> None:
    _safe_print("=" * 74)
    _safe_print(" КРЫМСКИЙ МОСТ 1v1 — ОПЕРАТОР FPV ДРОНА vs ВОДИТЕЛЬ-СТРЕЛОК БЕНЗОВОЗА")
    _safe_print(" Настольная 3D-игра для ПК  |  сетевой мультиплеер 1 на 1 по TCP")
    _safe_print("=" * 74)


def _check_dependencies() -> None:
    """Give an actionable message instead of a raw ImportError traceback."""
    missing = []
    for mod, hint in (
        ("pygame", "pygame-ce"),
        ("numpy", "numpy"),
    ):
        try:
            __import__(mod)
        except Exception:
            missing.append(hint)
    if missing:
        msg = (
            "Не установлены зависимости: " + ", ".join(missing) + "\n\n"
            "Выполните в папке игры:\n"
            "    python -m pip install -r requirements.txt\n\n"
            "Либо просто запустите run_windows.bat (он всё установит сам)."
        )
        _safe_print("[ОШИБКА] " + msg)
        raise SystemExit(2)


# -----------------------------------------------------------------------------
# Game modes
# -----------------------------------------------------------------------------
def run_headless_server(host: str, port: int, bot_fill: bool, auto_net: bool = True) -> int:
    from src.config import WEATHER_BY_ID
    from src.network import GameServer, describe_connection_targets

    srv = GameServer(host=host, port=port, fill_with_bot=bot_fill)
    bound = srv.start()
    targets = describe_connection_targets(bound)
    _safe_print("=" * 74)
    _safe_print(" ВЫДЕЛЕННЫЙ СЕРВЕР «КРЫМСКИЙ МОСТ 1v1» ЗАПУЩЕН!")
    _safe_print(f" Слушает адрес:  {host}:{bound}")
    _safe_print(" Что вписать другу в поле «IP» (выберите тот адрес, по которому он подключается):")
    for t in targets:
        _safe_print(f"    -> {t}")
    _safe_print("   (192.168.x.x — обычная домашняя сеть, работает в одной Wi-Fi сети)")
    _safe_print(f" Заполнение пустого слота ИИ-ботом: {'ВКЛ' if bot_fill else 'ВЫКЛ'}")
    _safe_print("=" * 74)

    # --- Автоматическая настройка доступа извне -----------------------------
    auto_host = None
    if auto_net:
        try:
            from src.net_setup import AutoHostSetup

            def _on_update(snapshot: dict) -> None:
                status = snapshot.get("status", "")
                addrs = snapshot.get("addresses", [])
                if addrs:
                    _safe_print("")
                    _safe_print(" ГОТОВЫЙ АДРЕС ДЛЯ ДРУГА (скопируйте ему):")
                    for label, address in addrs:
                        _safe_print(f"    -> {address}    ({label})")
                    _safe_print("   Друг вставляет этот адрес в поле «IP» и жмёт ENTER.")
                elif status:
                    _safe_print(f" [настройка доступа] {status}")

            auto_host = AutoHostSetup(port=bound, on_update=_on_update)
            auto_host.start()
            _safe_print(" Настраиваем доступ из интернета (UPnP / бесплатный туннель)...")
        except Exception as exc:  # noqa: BLE001
            _safe_print(f" [настройка доступа] не удалось: {exc}")

    _safe_print(" Нажмите Ctrl+C для остановки сервера.")
    _safe_print("=" * 74)

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
            _safe_print(
                f"\r[Раунд {snap['round_number']} | {snap['phase']}] "
                f"Счёт FPV {snap['drone_score']}:{snap['truck_score']} Бензовоз | "
                f"Погода: {weather_name} | "
                f"Игроки: FPV={'ОНЛАЙН' if d_conn else '---'}, Бензовоз={'ОНЛАЙН' if t_conn else '---'}   "
            )
            time.sleep(1.0)
    finally:
        _safe_print("\nОстановка сервера...")
        if auto_host is not None:
            try:
                auto_host.stop()
            except Exception:
                pass
        srv.stop()
    return 0


def run_cli_self_test(repo_root: str, screenshots_dir: str | None) -> int:
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from src.selftest import run_verification_suite

    _safe_print("=" * 76)
    _safe_print(" ЗАПУСК ПРОВЕРОЧНОГО РЕЖИМА («Крымский Мост 1v1 — FPV vs Бензовоз»)")
    _safe_print("=" * 76)

    def _printer(res) -> None:
        badge = "[ OK ]" if res.passed else "[FAIL]"
        _safe_print(f" {badge} {res.name_ru}")
        _safe_print(f"        └─ {res.details_ru} ({res.duration_ms:.1f} мс)")

    results = run_verification_suite(
        repo_root=repo_root,
        save_screenshots_dir=screenshots_dir,
        on_progress=_printer,
    )
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    _safe_print("-" * 76)
    if passed == total:
        _safe_print(
            f" ИТОГ: ВСЕ ПРОВЕРКИ УСПЕШНО ПРОЙДЕНЫ ({passed}/{total})! "
            "ИГРА ПОЛНОСТЬЮ РАБОТОСПОСОБНА."
        )
        return 0
    _safe_print(f" ИТОГ: ОБНАРУЖЕНЫ ОШИБКИ ({passed}/{total} пройдено).")
    return 1


def run_desktop_gui(repo_root: str, width: int, height: int, start_mode: str) -> int:
    from src.app import DesktopGameApp

    app = DesktopGameApp(
        repo_root=repo_root,
        width=width,
        height=height,
        start_mode=start_mode,
    )
    app.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Настольная 3D-игра «Крымский Мост 1v1: FPV Дрон против Бензовоза»"
    )
    parser.add_argument("--server", action="store_true", help="Запустить выделенный сетевой сервер 1v1 (без 3D окна)")
    parser.add_argument("--host", default="0.0.0.0", help="IP-интерфейс сервера (по умолчанию 0.0.0.0)")
    parser.add_argument("--port", type=int, default=27015, help="TCP порт сервера (по умолчанию 27015)")
    parser.add_argument("--bot-fill", action="store_true", help="На сервере: подставлять ИИ-бота, пока подключён только 1 игрок")
    parser.add_argument("--no-auto-net", action="store_true",
                        help="На сервере: не пытаться автоматически открыть порт (UPnP) и поднять туннель")
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
        return run_headless_server(args.host, args.port, args.bot_fill, auto_net=not args.no_auto_net)

    return run_desktop_gui(repo_root, args.width, args.height, "verify" if args.verify else "menu")


def _entry_point() -> int:
    repo_root = get_repo_root()
    _prepare_streams(repo_root)
    _install_crash_handler(repo_root)

    try:
        _check_dependencies()
        _print_banner()
        return main()
    except SystemExit:
        raise
    except Exception:
        text = traceback.format_exc()
        log_path = _write_crash_log(repo_root, text)
        _safe_print("=" * 78)
        _safe_print("КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ИГРЫ")
        _safe_print("=" * 78)
        _safe_print(text)
        _safe_print(f"Отчёт сохранён в файле: {log_path}")
        _safe_print("=" * 78)
        _show_error_box(
            "Крымский Мост 1v1 — ошибка запуска",
            f"Не удалось запустить игру.\n\n"
            f"Подробности:\n{log_path}\n\n"
            "Чаще всего помогает запуск run_windows.bat, который сам ставит зависимости.",
        )
        # Держим окно консоли открытым, чтобы пользователь успевал прочитать ошибку
        try:
            if sys.stdin is not None and sys.stdout is not None and sys.stdin.isatty():
                input("\nНажмите Enter, чтобы закрыть окно...")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_entry_point())
