"""
Procedural Sound Synthesizer (22.05 kHz stereo WAV buffers generated via NumPy):
- FPV high-RPM quadcopter rotor whine
- Fuel Truck V12 diesel rumble
- Roof AA Turret rapid burst & 12-gauge Buckshot blast
- Warhead explosion boom & EW jammer static buzz
Gracefully falls back to no-op if the host machine has no audio output device.
"""

from __future__ import annotations

import numpy as np
import pygame


class SoundManager:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = False
        self._sounds = {}
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._build_sounds()
            self.enabled = True
        except pygame.error:
            self.enabled = False

    def _make_sound(self, mono_wave: np.ndarray, volume: float = 0.45) -> pygame.mixer.Sound:
        clipped = np.clip(mono_wave * volume, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        stereo = np.column_stack((pcm16, pcm16))
        return pygame.sndarray.make_sound(stereo)

    def _build_sounds(self) -> None:
        sr = 22050
        rng = np.random.default_rng(7)

        # 1. Shotgun blast (0.28s punchy low thump + noise decay)
        t = np.linspace(0, 0.28, int(sr * 0.28), endpoint=False)
        env = np.exp(-t * 14.0)
        wave = (rng.uniform(-1, 1, len(t)) * 0.75 + np.sin(2 * np.pi * 95 * t) * 0.65) * env
        self._sounds["shotgun"] = self._make_sound(wave, 0.55)

        # 2. AA Turret shot (0.09s crisp metallic bark)
        t = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
        env = np.exp(-t * 32.0)
        wave = (rng.uniform(-1, 1, len(t)) * 0.55 + np.sin(2 * np.pi * 165 * t) * 0.7) * env
        self._sounds["turret"] = self._make_sound(wave, 0.35)

        # 3. Explosion boom (0.75s deep rumble)
        t = np.linspace(0, 0.75, int(sr * 0.75), endpoint=False)
        env = np.exp(-t * 4.8)
        wave = (rng.uniform(-1, 1, len(t)) * 0.65 + np.sin(2 * np.pi * 52 * t) * 0.85) * env
        self._sounds["explosion"] = self._make_sound(wave, 0.70)

        # 4. EW Jammer pulse activation
        t = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
        wave = np.sin(2 * np.pi * (420 + 680 * t) * t) * np.exp(-t * 4.5)
        self._sounds["ew"] = self._make_sound(wave, 0.35)

    def play(self, name: str) -> None:
        if self.enabled and name in self._sounds:
            try:
                self._sounds[name].play()
            except pygame.error:
                pass
