"""
Sound Engine for "Crimean Bridge 1v1".

Real CC0 recordings (assets/sfx/*.wav, see assets/sfx/CREDITS.md) are used for
weapons, impacts and interface. Seamless ambience loops (engine, rotors, wind,
rain, EW interference) are synthesised procedurally with NumPy, because loops
must be endless and tiny.

Features:
* distance model - every shot has a "near" and a "far" (low-passed) variant,
  volume falls off with the distance to the sound source;
* random pitch variation - three variants per sample so a burst of fire never
  sounds like a machine gun loop;
* graceful degradation - if the samples are missing or there is no audio device,
  the game keeps working with a small procedural fallback.
"""

from __future__ import annotations

import os
import random
import wave
from typing import Dict, List, Optional

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover - pygame is a hard dependency in practice
    pygame = None  # type: ignore

SR = 22050

# Sounds that get a low-passed "far" variant in assets/sfx
_FAR_VARIANTS = {"turret", "shotgun", "explosion", "truck_hit", "drone_down"}


class SoundManager:
    def __init__(self, enabled: bool = True, repo_root: Optional[str] = None) -> None:
        self.enabled = False
        self._sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._variants: Dict[str, List["pygame.mixer.Sound"]] = {}
        self._loops: Dict[str, "pygame.mixer.Channel"] = {}
        self._loop_sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._rng = random.Random()
        self._assets_dir = ""
        self.master_volume = 0.85

        if not enabled or pygame is None:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512)
            self._assets_dir = self._resolve_assets_dir(repo_root)
            self._load_samples()
            self._build_loops()
            self.enabled = True
        except Exception:
            self.enabled = False

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_assets_dir(repo_root: Optional[str]) -> str:
        from .assets import find_dir

        found = find_dir(os.path.join("assets", "sfx"), repo_root or "")
        return found or ""

    @staticmethod
    def _read_wav_mono(path: str) -> Optional[np.ndarray]:
        try:
            with wave.open(path, "rb") as w:
                if w.getsampwidth() != 2:
                    return None
                n_ch = w.getnchannels()
                frames = w.readframes(w.getnframes())
            a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if n_ch > 1:
                a = a.reshape(-1, n_ch).mean(axis=1)
            return a
        except Exception:
            return None

    @staticmethod
    def _pitch_shift(a: np.ndarray, factor: float) -> np.ndarray:
        """Сдвиг тона (ресемплинг с сохранением длительности через перекрытие)."""
        if factor <= 0.01 or a.size < 8:
            return a
        out_len = int(a.size / factor)
        idx = np.arange(out_len) * factor
        stretched = np.interp(idx, np.arange(a.size), a).astype(np.float32)
        if out_len >= a.size:
            return stretched
        # доберём хвост простым наложением, чтобы не обрезать реверберацию
        tail = a[out_len:]
        return np.concatenate([stretched[: out_len], tail * 0.6]).astype(np.float32)

    def _make_sound(self, mono_wave: np.ndarray, volume: float = 0.45) -> "pygame.mixer.Sound":
        clipped = np.clip(mono_wave * volume, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        stereo = np.column_stack((pcm16, pcm16))
        return pygame.sndarray.make_sound(stereo)

    def _load_samples(self) -> None:
        if not self._assets_dir or not os.path.isdir(self._assets_dir):
            self._build_fallback_sounds()
            return
        for fname in sorted(os.listdir(self._assets_dir)):
            if not fname.endswith(".wav"):
                continue
            name = fname[:-4]
            data = self._read_wav_mono(os.path.join(self._assets_dir, fname))
            if data is None or data.size < 16:
                continue
            # Revoice the existing CC0 recordings with a quieter low-frequency
            # layer, without introducing unlicensed assets or external downloads.
            if name in ("turret", "shotgun"):
                layer_name = "blast_small.wav" if name == "turret" else "explosion_far.wav"
                layer = self._read_wav_mono(os.path.join(self._assets_dir, layer_name))
                if layer is not None and layer.size:
                    data = data.copy()
                    n = min(data.size, layer.size)
                    data[:n] = np.clip(data[:n] * 0.83 + layer[:n] * 0.17, -1.0, 1.0)
            base = self._make_sound(data, 1.0)
            self._sounds[name] = base
            # три варианта тона: -6%, базовый, +6%
            self._variants[name] = [
                self._make_sound(self._pitch_shift(data, 1.06), 1.0),
                base,
                self._make_sound(self._pitch_shift(data, 0.94), 1.0),
            ]
        if "explosion" not in self._sounds:
            self._build_fallback_sounds()

    def _build_fallback_sounds(self) -> None:
        """Минимальный процедурный набор, если файлов с сэмплами нет."""
        sr = SR
        rng = np.random.default_rng(7)

        t = np.linspace(0, 0.28, int(sr * 0.28), endpoint=False)
        env = np.exp(-t * 14.0)
        wave_s = (rng.uniform(-1, 1, len(t)) * 0.75 + np.sin(2 * np.pi * 95 * t) * 0.65) * env
        self._sounds["shotgun"] = self._make_sound(wave_s, 0.55)

        t = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
        env = np.exp(-t * 32.0)
        wave_s = (rng.uniform(-1, 1, len(t)) * 0.55 + np.sin(2 * np.pi * 165 * t) * 0.7) * env
        self._sounds["turret"] = self._make_sound(wave_s, 0.35)

        t = np.linspace(0, 0.75, int(sr * 0.75), endpoint=False)
        env = np.exp(-t * 4.8)
        wave_s = (rng.uniform(-1, 1, len(t)) * 0.65 + np.sin(2 * np.pi * 52 * t) * 0.85) * env
        self._sounds["explosion"] = self._make_sound(wave_s, 0.70)

    # ------------------------------------------------------------------
    # Procedural seamless loops
    # ------------------------------------------------------------------
    def _build_loops(self) -> None:
        sr = SR
        rng = np.random.default_rng(19)

        def fade_loop(x: np.ndarray, fade: int = 900) -> np.ndarray:
            """Склейка начала и конца с плавным переходом -> бесшовный цикл."""
            if x.size < fade * 2:
                return x
            out = x.copy()
            ramp = np.linspace(0.0, 1.0, fade)
            out[:fade] = out[:fade] * ramp + x[-fade:] * (1.0 - ramp)
            return out[:-fade]

        # 1. Дизель V12 (тон + шум + медленная детонация)
        n = sr * 2
        t = np.arange(n) / sr
        base = 46.0
        diesel = (
            np.sin(2 * np.pi * base * t)
            + 0.55 * np.sin(2 * np.pi * base * 2 * t)
            + 0.28 * np.sin(2 * np.pi * base * 3 * t + 0.6)
        )
        rough = np.cumsum(rng.normal(0, 1, n))
        rough = rough / max(np.abs(rough).max(), 1e-6)
        fire = 0.5 + 0.5 * np.sin(2 * np.pi * 7.5 * t)
        engine = (diesel * (0.45 + 0.25 * fire) + rough * 0.16 * fire)
        self._loop_sounds["engine"] = self._make_sound(fade_loop(engine), 0.30)

        # 2. Визг моторов FPV (высокая частота + амплитудная модуляция)
        n = sr * 1
        t = np.arange(n) / sr
        rotor = np.sin(2 * np.pi * 520 * t) * 0.28 + np.sin(2 * np.pi * 1040 * t) * 0.15
        rotor *= 0.78 + 0.22 * np.sin(2 * np.pi * 18 * t)
        rotor += rng.normal(0, 0.05, n)
        self._loop_sounds["rotor"] = self._make_sound(fade_loop(rotor), 0.22)

        # 3. Ветер (отфильтрованный шум с медленным LFO)
        n = sr * 3
        noise = rng.normal(0, 1, n)
        kernel = np.ones(24) / 24.0
        wind = np.convolve(noise, kernel, mode="same")
        wind = wind / max(np.abs(wind).max(), 1e-6)
        t = np.arange(n) / sr
        wind *= 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.23 * t))
        self._loop_sounds["wind"] = self._make_sound(fade_loop(wind, 2500), 0.20)

        # 4. Дождь / снег (плотный шум + редкие капли)
        n = sr * 2
        rain = rng.normal(0, 1, n)
        kernel = np.ones(6) / 6.0
        rain = np.convolve(rain, kernel, mode="same")
        rain /= max(np.abs(rain).max(), 1e-6)
        drops = np.zeros(n)
        for _ in range(90):
            pos = int(self._rng.random() * (n - 400))
            length = int(sr * 0.02)
            drops[pos : pos + length] += np.exp(-np.linspace(0, 5, length)) * self._rng.random()
        self._loop_sounds["rain"] = self._make_sound(fade_loop(rain * 0.8 + drops * 0.5, 1200), 0.18)

        # 5. Помехи РЭБ (шум + «дребезг» + warble)
        n = sr * 1
        t = np.arange(n) / sr
        static_noise = rng.normal(0, 1, n) * 0.6
        warble = np.sin(2 * np.pi * (320 + 140 * np.sin(2 * np.pi * 3.1 * t)) * t)
        ew = static_noise * 0.6 + warble * 0.35
        ew *= 0.65 + 0.35 * np.sin(2 * np.pi * 9.0 * t)
        self._loop_sounds["ew_static"] = self._make_sound(fade_loop(ew, 700), 0.16)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def play(self, name: str, distance: Optional[float] = None, volume: float = 1.0) -> None:
        """Проигрывает звук. `distance` (м) включает затухание и «дальний» вариант."""
        if not self.enabled:
            return
        try:
            snd = self._sounds.get(name)
            if snd is None:
                return
            gain = self.master_volume * volume
            if distance is not None and distance > 0.0:
                # затухание в воздухе + переход на «дальний» тембр
                far_name = name + "_far"
                if far_name in self._sounds and distance > 55.0:
                    snd = self._sounds[far_name]
                gain *= 1.0 / (1.0 + distance / 55.0)
            # Use the actual near/far sample for pitch variations.
            variant_name = far_name if distance is not None and distance > 55.0 and far_name in self._sounds else name
            variants = self._variants.get(variant_name)
            if variants:
                snd = variants[self._rng.randrange(len(variants))]
            snd.set_volume(float(max(0.0, min(1.0, gain))))
            snd.play()
        except pygame.error:
            pass

    def start_loop(self, name: str, volume: float = 0.5) -> None:
        if not self.enabled:
            return
        snd = self._loop_sounds.get(name)
        if snd is None:
            return
        try:
            if name in self._loops:
                self._loops[name].set_volume(float(max(0.0, min(1.0, volume))))
                return
            snd.set_volume(float(max(0.0, min(1.0, volume))))
            self._loops[name] = snd.play(loops=-1)
        except pygame.error:
            pass

    def stop_loop(self, name: str) -> None:
        ch = self._loops.pop(name, None)
        if ch is not None:
            try:
                ch.stop()
            except pygame.error:
                pass

    def stop_all_loops(self) -> None:
        for name in list(self._loops.keys()):
            self.stop_loop(name)

    def set_loop_volume(self, name: str, volume: float) -> None:
        ch = self._loops.get(name)
        if ch is not None:
            try:
                ch.set_volume(float(max(0.0, min(1.0, volume))))
            except pygame.error:
                pass

    def has(self, name: str) -> bool:
        return name in self._sounds
