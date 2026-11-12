#!/usr/bin/env python3
"""
Сборка игровых звуков из CC0-источников.

Источники (скачиваются вручную, публичные, лицензия CC0 1.0):
  * npm-пакет `@netforge/asset-library-files`  -> боевые звуки (выстрел, попадания, взрывы)
  * npm-пакет `uisfx`                            -> интерфейсные звуки (11 стилей)

Скрипт:
  1. декодирует OGG/MP3 через SDL2-mixer (pygame),
  2. обрезает тишину по краям,
  3. нормализует громкость (пик + лимитер),
  4. делает «дальний» вариант (ФНЧ-фильтр + реверберационный хвост) для звуков издалека,
  5. сохраняет 16-битные моно WAV 22050 Гц в assets/sfx/.

Запуск:
    python tools/build_sfx.py --netforge /tmp/sfx/netforge --uisfx /tmp/sfx/uisfx/sounds
"""

from __future__ import annotations

import argparse
import os
import sys
import wave

# На серверах без звуковой карты SDL2-mixer не инициализируется -> используем dummy-драйвер
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

SR = 22050


# ---------------------------------------------------------------------------
# Утилиты обработки
# ---------------------------------------------------------------------------
def decode(path: str) -> np.ndarray:
    """Декодирует звук (OGG/MP3) в моно float32 [-1..1] на частоте SR.

    Используется soundfile (libsndfile) — он надёжно читает Ogg Vorbis,
    в отличие от pygame.mixer.get_raw() для сжатых потоков.
    """
    import soundfile as sf

    data, src_sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return resample_to(mono, int(src_sr), SR)


def resample_to(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Ресемплинг с антиалиасинговым ФНЧ (FIR, взвешенное окно Хэмминга)."""
    if src_sr == dst_sr or x.size == 0:
        return x
    ratio = dst_sr / src_sr
    cutoff = 0.45 * dst_sr
    n_taps = 63
    n = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = (2.0 * cutoff / src_sr) * np.sinc((2.0 * cutoff / src_sr) * n) * np.hamming(n_taps)
    filtered = np.convolve(x, h, mode="same")
    out_len = max(1, int(x.size * ratio))
    idx = np.linspace(0.0, filtered.size - 1.0, out_len)
    return np.interp(idx, np.arange(filtered.size), filtered).astype(np.float32)


def trim_silence(a: np.ndarray, threshold: float = 0.006) -> np.ndarray:
    """Убирает тишину в начале/конце, оставляя небольшой хвост."""
    if a.size == 0:
        return a
    loud = np.abs(a) > threshold
    if not loud.any():
        return a
    idx = np.flatnonzero(loud)
    start = max(0, idx[0] - int(SR * 0.004))
    end = min(a.size, idx[-1] + int(SR * 0.06))
    return a[start:end]


def normalize(a: np.ndarray, peak: float = 0.92, rms_target: float = 0.11) -> np.ndarray:
    """Пиковая нормализация с мягким лимитером и подгонкой по RMS."""
    if a.size == 0:
        return a
    m = float(np.abs(a).max())
    if m > 1e-6:
        a = a / m
    rms = float(np.sqrt((a ** 2).mean()))
    if rms > 1e-6:
        gain = rms_target / rms
        a = a * float(np.clip(gain, 0.35, 3.2))
    # мягкий лимитер (tanh)
    a = np.tanh(a * 1.25) / np.tanh(1.25)
    m = float(np.abs(a).max())
    if m > 1e-6:
        a = a * (peak / m)
    return np.clip(a, -1.0, 1.0)


def one_pole_lowpass(a: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Однополюсный ФНЧ — «звук издалека» (глуше, без верхних частот)."""
    if a.size == 0:
        return a
    x = math_exp = np.exp(-2.0 * np.pi * cutoff_hz / SR)
    y = np.empty_like(a)
    acc = 0.0
    for i, v in enumerate(a):
        acc = (1.0 - x) * v + x * acc
        y[i] = acc
    return y


def make_far(a: np.ndarray, cutoff_hz: float = 1100.0, tail: float = 0.22, gain: float = 0.55) -> np.ndarray:
    """Дальний вариант: ФНЧ + ослабление + короткий реверберационный хвост."""
    lp = one_pole_lowpass(a, cutoff_hz) * gain
    n_tail = int(SR * tail)
    if n_tail > 0 and lp.size > 8:
        # простая «реверберация»: свёртка с экспоненциальным шумом
        rng = np.random.default_rng(11)
        ir = rng.uniform(-1.0, 1.0, n_tail) * np.exp(-np.linspace(0.0, 5.0, n_tail))
        ir[0] = 1.0
        wet = np.convolve(lp, ir, mode="full")[: lp.size + n_tail]
        lp = np.concatenate([lp, wet[lp.size:]])
    return lp


def write_wav(path: str, a: np.ndarray) -> None:
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Таблица: имя в игре <- источник
# ---------------------------------------------------------------------------
# (game_name, source_kind, source_file, gain, rms, far_variant)
SFX_TABLE = [
    # --- Оружие и попадания (netforge, CC0) ---
    ("turret",       "netforge", "shoot.ogg",     0.85, 0.16, True),
    ("shotgun",      "netforge", "impact.ogg",   0.95, 0.17, True),
    ("explosion",    "netforge", "big_kill.ogg", 0.95, 0.19, True),
    ("blast_small",  "netforge", "hit.ogg",      0.85, 0.14, False),
    ("truck_hit",    "netforge", "hurt.ogg",     0.85, 0.15, True),
    ("drone_hit",    "netforge", "hit.ogg",      0.80, 0.13, False),
    ("drone_down",   "netforge", "downed.ogg",   0.90, 0.16, True),
    ("reload",       "netforge", "dash.ogg",     0.70, 0.12, False),
    ("drone_start",  "netforge", "revive.ogg",   0.80, 0.14, False),
    ("land",         "netforge", "land.ogg",     0.75, 0.13, False),
    ("step",         "netforge", "step.ogg",     0.55, 0.07, False),
    # --- Интерфейс (uisfx, CC0) ---
    ("ui_click",     "uisfx", "mechanical/press.ogg",       0.55, 0.10, False),
    ("ui_move",      "uisfx", "minimal/hover.ogg",          0.45, 0.07, False),
    ("ui_confirm",   "uisfx", "minimal/select.ogg",         0.60, 0.12, False),
    ("ui_error",     "uisfx", "mechanical/error.ogg",       0.65, 0.13, False),
    ("ui_pick",      "uisfx", "mechanical/toggle-on.ogg",   0.60, 0.12, False),
    ("ui_notify",    "uisfx", "minimal/notification.ogg",   0.60, 0.12, False),
    ("ui_warning",   "uisfx", "mechanical/warning.ogg",     0.65, 0.13, False),
    ("round_start",  "uisfx", "scifi/start.ogg",            0.70, 0.14, False),
    ("victory",      "uisfx", "minimal/success.ogg",        0.70, 0.14, False),
    ("defeat",       "uisfx", "mechanical/blocked.ogg",     0.70, 0.14, False),
    ("ew_on",        "uisfx", "scifi/connect.ogg",           0.70, 0.13, False),
    ("ew_loop",      "uisfx", "scifi/processing.ogg",        0.45, 0.08, False),
    ("smoke",        "uisfx", "mechanical/expand.ogg",      0.55, 0.09, False),
]

CREDITS = """# Источники звуков

Все звуки в этой папке — **публичное достояние (CC0 1.0 Universal)**.
Можно свободно использовать, изменять и распространять, атрибуция не требуется
(но мы её указываем из уважения к авторам).

| Игровое имя | Источник | Оригинал | Лицензия |
|---|---|---|---|
| `turret`, `shotgun`, `explosion`, `blast_small`, `truck_hit`, `drone_hit`, `drone_down`, `reload`, `drone_start`, `land`, `step` | npm `@netforge/asset-library-files` (каталог `library/audio/sfx`) | `shoot.ogg`, `impact.ogg`, `big_kill.ogg`, `hit.ogg`, `hurt.ogg`, `downed.ogg`, `dash.ogg`, `revive.ogg`, `land.ogg`, `step.ogg` | CC0 1.0 |
| `ui_*`, `round_start`, `victory`, `defeat`, `ew_on`, `ew_loop`, `smoke` | npm `uisfx` (папки `mechanical`, `minimal`, `scifi`) | `click.ogg`, `hover.ogg`, `confirm.ogg`, `error.ogg`, `toggle-on.ogg`, `notification.ogg`, `warning.ogg`, `start.ogg`, `success.ogg`, `blocked.ogg`, `connect.ogg`, `processing.ogg`, `expand.ogg` | CC0 1.0 |

Пересборка: `python tools/build_sfx.py --netforge <каталог> --uisfx <каталог>`
Обработка: обрезка тишины, нормализация громкости, «дальний» вариант (ФНЧ + хвост),
16-битное моно 22050 Гц.

Процедурные (синтезируются в `src/audio.py`, файлов на диске нет):
ветер, шум дождя/снега, «свист» РЭБ-помех, гул двигателя V12, визг моторов FPV.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--netforge", required=True, help="каталог с .ogg из @netforge/asset-library-files")
    ap.add_argument("--uisfx", required=True, help="каталог со стилями из uisfx/sounds")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sfx"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    made = []
    for name, kind, fname, gain, rms, far in SFX_TABLE:
        src = os.path.join(args.netforge if kind == "netforge" else args.uisfx, fname)
        if not os.path.isfile(src):
            print(f"  ПРОПУЩЕНО (нет файла): {name} <- {src}")
            continue
        a = decode(src)
        a = trim_silence(a)
        a = normalize(a, peak=gain, rms_target=rms)
        out_path = os.path.join(args.out, name + ".wav")
        write_wav(out_path, a)
        size = os.path.getsize(out_path)
        total += size
        made.append((name, a.size / SR, size))
        print(f"  {name:14s} {a.size / SR:5.2f}s  {size / 1024:6.1f} KB")

        if far:
            far_a = make_far(a)
            far_path = os.path.join(args.out, name + "_far.wav")
            write_wav(far_path, far_a)
            fs = os.path.getsize(far_path)
            total += fs
            made.append((name + "_far", far_a.size / SR, fs))
            print(f"  {name + '_far':14s} {far_a.size / SR:5.2f}s  {fs / 1024:6.1f} KB")

    with open(os.path.join(args.out, "CREDITS.md"), "w", encoding="utf-8") as f:
        f.write(CREDITS)

    print(f"\nГотово: {len(made)} файлов, суммарно {total / 1024:.1f} КБ -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
