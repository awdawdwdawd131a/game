"""
3D Perspective Scene Renderer + Role-Specific Tactical HUDs:
1. FPV Drone View:
   - True 6-DOF rolled/pitched 110° wide-angle FPV camera
   - Analog OSD overlay: Artificial Horizon, Voltage/Battery, RSSI/EW Jamming static snow,
     Wind vector indicator, Target Box (Thermal highlight if Airburst/Thermal perk active)
2. Fuel Truck Driver/Gunner View:
   - Chase/Roof AA Gunner 3D camera (rotates 360° around cab roof so player can drive forward
     while tracking & shooting down the diving FPV drone behind/above!)
   - Rear-view/Drone-Threat Radar compass, Speedometer, Cistern Integrity HP, Turret Heat bar,
     Shotgun reload, EW Jammer & Smoke cooldowns, Bridge Finish progress bar.
"""

from __future__ import annotations

import math
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from .config import (
    BRIDGE_FINISH_Z,
    BRIDGE_START_Z,
    DRONE_MOD_BY_ID,
    ROAD_DECK_Y,
    TRUCK_MOD_BY_ID,
    WeatherPreset,
)
from .math3d import (
    Mesh3D,
    build_crimean_bridge_mesh,
    build_fpv_drone_mesh,
    build_fuel_truck_mesh,
    build_turret_mesh,
    clamp,
    euler_to_matrix,
    forward_from_angles,
)
from .physics import WorldSimulation


@dataclass
class Camera3D:
    pos: np.ndarray
    yaw: float
    pitch: float
    roll: float
    fov_deg: float = 82.0


class SceneRenderer3D:
    def __init__(self, width: int, height: int, barriers: list, repo_root: str) -> None:
        self.width = width
        self.height = height
        self.repo_root = repo_root

        # Prebuild static & dynamic 3D meshes
        self.bridge_mesh, _ = build_crimean_bridge_mesh(barriers)
        self.truck_meshes: Dict[bool, Mesh3D] = {
            False: build_fuel_truck_mesh(has_cage=False),
            True: build_fuel_truck_mesh(has_cage=True),
        }
        self.turret_meshes: Dict[bool, Mesh3D] = {
            False: build_turret_mesh(twin=False),
            True: build_turret_mesh(twin=True),
        }
        self.drone_meshes: Dict[bool, Mesh3D] = {
            False: build_fpv_drone_mesh(shaped_charge=False),
            True: build_fpv_drone_mesh(shaped_charge=True),
        }

        # Precompute sun light direction
        sun = np.array([-0.42, 0.78, -0.46], dtype=np.float32)
        self.sun_dir = sun / float(np.linalg.norm(sun))

        # Fonts (Cyrillic-capable system DejaVu / Arial / fallback)
        pygame.font.init()
        self.font_xs = self._load_font(14)
        self.font_sm = self._load_font(17)
        self.font_md = self._load_font(22, bold=True)
        self.font_lg = self._load_font(34, bold=True)
        self.font_xl = self._load_font(48, bold=True)

        # Optional Easter-egg emblem from existing `черешня.jpg` (read-only! Never modified!)
        self.easter_egg_thumb: Optional[pygame.Surface] = self._load_chereshnya_thumb()
        self._rng = np.random.default_rng(99)

        # --- Производительность: внутренний масштаб рендера 3D-сцены -------------
        # 1.0 = полное разрешение, 0.6 = 60% (сильно быстрее на слабых ПК).
        self.render_scale = 1.0
        self._world_surface: Optional[pygame.Surface] = None
        self._world_size = (0, 0)

        # --- Небо: звёзды, луна, солнце, облака, огни города ---------------------
        self._stars = self._make_stars(340)
        self._clouds = self._make_clouds(96)
        self._city_lights = self._make_city_lights(120)
        self._sky_time = 0.0
        self._cloud_sprites = self._make_cloud_sprite()
        self._glow_sprite = self._make_glow_sprite()
        self._sun_sprite = self._make_sun_disc_sprite()
        self._tint_cache: Dict[Tuple[int, int, int], pygame.Surface] = {}
        self._cloud_cache: Dict[tuple, pygame.Surface] = {}
        self._sky_buf: Optional[pygame.Surface] = None
        self._sky_up: Optional[pygame.Surface] = None
        self._cloud_layer: Optional[pygame.Surface] = None
        self._glow_cache: Dict[tuple, np.ndarray] = {}

        # --- Адаптивный render-scale ----------------------------------------
        # Мир рисуется во внутреннем буфере пониженного разрешения и затем
        # растягивается на экран. На слабых машинах это заметно ускоряет
        # растеризацию полигонов, а HUD остаётся чётким (он рисуется
        # поверх, в полном разрешении).
        self.render_scale = 1.0          # текущий масштаб мира
        self._render_scale_target = 1.0  # желаемый масштаб (1.0 = native)
        self._frame_ms_avg = 0.0         # сглаженное время кадра
        self._scale_cooldown = 0.0       # кадры до следующей подстройки
        self._world_buf: Optional[pygame.Surface] = None
        self._world_up: Optional[pygame.Surface] = None

    # ------------------------------------------------------------------
    # Небесные детали (генерируются один раз)
    # ------------------------------------------------------------------
    def _make_stars(self, count: int) -> np.ndarray:
        """Звёзды как направления на небесной сфере: (азимут, высота, яркость, размер)."""
        rng = np.random.default_rng(4242)
        stars = np.zeros((count, 4), dtype=np.float32)
        stars[:, 0] = rng.random(count) * math.tau          # азимут
        # высота: смещено к зениту, у горизонта гаснут
        stars[:, 1] = 0.04 + (rng.random(count) ** 0.65) * 1.15
        stars[:, 2] = 0.30 + rng.random(count) * 0.70
        stars[:, 3] = 1.0 + (rng.random(count) < 0.12) * 1.6
        return stars

    def _make_clouds(self, count: int) -> np.ndarray:
        """Облака как направления на небесной сфере: (азимут, высота, размер, плотность)."""
        rng = np.random.default_rng(777)
        clouds = np.zeros((count, 5), dtype=np.float32)
        clouds[:, 0] = rng.random(count) * math.tau
        # Высота над горизонтом: облака держатся у горизонта (как у моря),
        # поэтому распределение смещено к низу неба.
        clouds[:, 1] = 0.02 + (rng.random(count) ** 2.0) * 0.46
        # УГЛОВОЙ размер облака в радианах (0.035..0.20 рад = 2..11 градусов).
        # Раньше здесь была доля от ширины экрана - из-за этого облако
        # растягивалось на треть кадра и выглядело размытым пятном.
        clouds[:, 2] = 0.035 + (rng.random(count) ** 1.5) * 0.165
        clouds[:, 3] = 0.35 + rng.random(count) * 0.65
        clouds[:, 4] = rng.random(count) * math.tau        # фаза для разброса по «глубине»
        return clouds

    def _make_city_lights(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(31337)
        lights = rng.random((count, 3)).astype(np.float32)
        lights[:, 2] = 0.5 + rng.random(count) * 0.5
        return lights

    # ------------------------------------------------------------------
    # Спрайты: мягкое облако и радиальное свечение
    # ------------------------------------------------------------------
    def _make_cloud_sprite(self) -> List[pygame.Surface]:
        """
        Мягкие «пуховые» облака (три варианта формы).

        Альфа-канал - сумма гауссовых лепестков: так не появляется
        прямоугольных артефактов, которые давали ellipse+BLEND_RGBA_MAX.
        RGB содержит вертикальное затенение (свет сверху, тень снизу),
        поэтому облако выглядит объёмным, а не плоским пятном.
        """
        sw, sh = 320, 150
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        layouts = [
            [
                (62.0, 84.0, 78.0, 34.0, 1.00),
                (148.0, 62.0, 92.0, 44.0, 1.15),
                (232.0, 80.0, 80.0, 36.0, 1.00),
                (196.0, 100.0, 72.0, 28.0, 0.85),
                (96.0, 104.0, 66.0, 26.0, 0.80),
                (258.0, 96.0, 54.0, 24.0, 0.72),
            ],
            [
                (52.0, 70.0, 62.0, 30.0, 0.95),
                (108.0, 58.0, 70.0, 40.0, 1.05),
                (168.0, 76.0, 86.0, 38.0, 1.00),
                (238.0, 66.0, 66.0, 32.0, 0.92),
                (286.0, 88.0, 58.0, 26.0, 0.78),
                (196.0, 104.0, 70.0, 24.0, 0.70),
            ],
            [
                (88.0, 78.0, 74.0, 36.0, 1.00),
                (166.0, 56.0, 84.0, 42.0, 1.12),
                (244.0, 74.0, 74.0, 34.0, 0.98),
                (126.0, 100.0, 62.0, 26.0, 0.82),
                (208.0, 98.0, 60.0, 24.0, 0.74),
            ],
        ]
        sprites: List[pygame.Surface] = []
        for blobs in layouts:
            alpha = np.zeros((sh, sw), dtype=np.float32)
            for bx, by, rx, ry, amp in blobs:
                d = ((xx - bx) / rx) ** 2 + ((yy - by) / ry) ** 2
                alpha += amp * np.exp(-d * 1.9)
            alpha = np.clip(alpha, 0.0, 1.0)
            # Плавное затухание к краям спрайта, чтобы не было видно «рамки»
            edge = np.minimum(
                np.minimum(xx, sw - 1.0 - xx) / (sw * 0.18),
                np.minimum(yy, sh - 1.0 - yy) / (sh * 0.18),
            )
            alpha *= np.clip(edge, 0.0, 1.0)
            # Вертикальное затенение: верх светлее, низ темнее (объём)
            shade = 1.0 - 0.38 * np.clip(yy / float(sh - 1), 0.0, 1.0)
            rgba = np.zeros((sh, sw, 4), dtype=np.uint8)
            for ch in range(3):
                rgba[:, :, ch] = np.clip(255.0 * shade, 0, 255).astype(np.uint8)
            rgba[:, :, 3] = (alpha * 235.0).astype(np.uint8)
            sprites.append(pygame.image.frombuffer(rgba.tobytes(), (sw, sh), "RGBA"))
        return sprites

    def _make_sun_disc_sprite(self) -> pygame.Surface:
        """
        Диск светила: плотное ядро с горячим центром и мягким краем.
        Рисуется аддитивно (BLEND_RGB_ADD), поэтому в центре получается
        белое ядро, а по краям - цвет светила.
        """
        r = 64
        yy, xx = np.mgrid[0:r * 2, 0:r * 2].astype(np.float32)
        d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / float(r)
        # Плотное ядро до 0.72 радиуса, затем быстрое затухание
        core = np.clip((0.78 - d) / 0.14, 0.0, 1.0)
        halo = np.clip((1.0 - d) / 0.30, 0.0, 1.0) ** 2.0
        alpha = np.clip(core + halo * 0.55, 0.0, 1.0)
        # Цвет: ядро белое (перегрев), к краю - цвет светила
        heat = np.clip((0.62 - d) / 0.34, 0.0, 1.0)
        rgba = np.zeros((r * 2, r * 2, 4), dtype=np.float32)
        for ch in range(3):
            rgba[:, :, ch] = 255.0 * (heat * 0.85 + (1.0 - heat) * 1.0)
        rgba[:, :, 3] = alpha * 255.0
        return pygame.image.frombuffer(
            rgba.astype(np.uint8).tobytes(), (r * 2, r * 2), "RGBA"
        )

    def _make_glow_sprite(self) -> pygame.Surface:
        """Радиальное свечение (ореол) для солнца/луны."""
        r = 128
        yy, xx = np.mgrid[0:r * 2, 0:r * 2].astype(np.float32)
        d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / float(r)
        alpha = np.clip(1.0 - d, 0.0, 1.0) ** 2.4
        rgba = np.zeros((r * 2, r * 2, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255
        rgba[:, :, 1] = 255
        rgba[:, :, 2] = 255
        rgba[:, :, 3] = (alpha * 255.0).astype(np.uint8)
        return pygame.image.frombuffer(rgba.tobytes(), (r * 2, r * 2), "RGBA")

    def _tinted(self, sprite: pygame.Surface, color: Tuple[int, int, int]) -> pygame.Surface:
        """Кэшированная тонировка спрайта (умножение RGB на цвет)."""
        key = (int(color[0]), int(color[1]), int(color[2]))
        cached = self._tint_cache.get(key)
        if cached is None:
            cached = sprite.copy()
            cached.fill((key[0], key[1], key[2], 255), special_flags=pygame.BLEND_RGBA_MULT)
            self._tint_cache[key] = cached
        return cached

    def _premultiplied(
        self,
        sprite: pygame.Surface,
        color: Tuple[int, int, int],
        intensity: float = 1.0,
    ) -> pygame.Surface:
        """
        Кэшированный спрайт с ПРЕДУМНОЖЕННЫМ цветом (RGB * alpha/255).
        Нужен для аддитивного блендинга (BLEND_RGB_ADD): иначе альфа-градиент
        игнорируется и свечение превращается в прямоугольник.

        `intensity` (0..1) - общая яркость свечения: слабое солнце в тумане
        не должно заливать пол-неба.
        """
        ikey = max(0, min(16, int(round(intensity * 16.0))))
        key = (-1, int(color[0]), int(color[1]), int(color[2]), ikey)
        cached = self._tint_cache.get(key)
        if cached is None:
            arr = pygame.surfarray.pixels_alpha(sprite).astype(np.float32)
            rgb = np.array(color, dtype=np.float32)[None, None, :] * (ikey / 16.0)
            out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
            out[:, :, :3] = np.clip(rgb * (arr[:, :, None] / 255.0), 0, 255).astype(np.uint8)
            out[:, :, 3] = 0
            cached = pygame.image.frombuffer(out.tobytes(), (arr.shape[1], arr.shape[0]), "RGBA")
            self._tint_cache[key] = cached
        return cached

    @staticmethod
    def _clip_near_plane(
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        f_cols: np.ndarray,
        f_norms: np.ndarray,
        f_two_sided: np.ndarray,
        near_z: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Разрезает треугольники, пересекающие ближнюю плоскость z = near_z.

        Треугольник может дать 1 или 2 новых треугольника. Реализовано
        векторно: для каждой грани считаются точки пересечения рёбер с
        плоскостью, затем собираются новые треугольники.

        Без этого клиппинга грани, проходящие через плоскость камеры,
        исчезали целиком - вблизи камеры (например, полотно моста под
        дроном) пропадали большие куски геометрии.
        """
        z0 = v0[:, 2]
        z1 = v1[:, 2]
        z2 = v2[:, 2]
        d0 = z0 - near_z
        d1 = z1 - near_z
        d2 = z2 - near_z

        # Классификация: сколько вершин «за камерой»
        behind = (d0 < 0).astype(np.int8) + (d1 < 0).astype(np.int8) + (d2 < 0).astype(np.int8)

        # Точка пересечения ребра (a -> b) с плоскостью
        def _cut(a: np.ndarray, b: np.ndarray, da: np.ndarray, db: np.ndarray) -> np.ndarray:
            denom = db - da
            t = np.where(np.abs(denom) > 1e-9, -da / np.where(np.abs(denom) > 1e-9, denom, 1.0), 0.0)
            return a + (b - a) * t[:, None]

        new_v0: List[np.ndarray] = []
        new_v1: List[np.ndarray] = []
        new_v2: List[np.ndarray] = []
        new_cols: List[np.ndarray] = []
        new_norms: List[np.ndarray] = []
        new_ts: List[np.ndarray] = []

        # --- 1 вершина за камерой: остаются 2 -> видимая часть - ЧЕТЫРЁХУГОЛЬНИК
        #      p_ab -> vb -> vc -> p_ca, который режется на два треугольника.
        #
        # Раньше здесь собирался ОДИН треугольник (p_ab, va, p_ca) с вершиной
        # va, лежащей ЗА камерой: из-за этого у моста под дроном появлялись
        # дыры и вытянутые «лоскуты» геометрии. Теперь оба треугольника
        # лежат строго перед плоскостью, а обход вершин сохраняет исходную
        # ориентацию грани (важно для backface-отсечения).
        m1 = behind == 1
        if np.any(m1):
            idx = np.nonzero(m1)[0]
            # Индекс единственной вершины за камерой (первая True в столбце)
            i0 = np.argmax(np.stack([d0, d1, d2]) < 0.0, axis=0)[idx]
            i1 = (i0 + 1) % 3
            i2 = (i0 + 2) % 3
            va = np.stack([v0, v1, v2], axis=1)[idx, i0]
            da = np.stack([d0, d1, d2], axis=1)[idx, i0]
            vb = np.stack([v0, v1, v2], axis=1)[idx, i1]
            vc = np.stack([v0, v1, v2], axis=1)[idx, i2]
            db = np.stack([d0, d1, d2], axis=1)[idx, i1]
            dc = np.stack([d0, d1, d2], axis=1)[idx, i2]
            p_ab = _cut(va, vb, da, db)
            p_ca = _cut(vc, va, dc, da)
            # Треугольник 1: p_ab -> vb -> vc
            new_v0.append(p_ab)
            new_v1.append(vb)
            new_v2.append(vc)
            # Треугольник 2: p_ab -> vc -> p_ca
            new_v0.append(p_ab)
            new_v1.append(vc)
            new_v2.append(p_ca)
            new_cols.append(f_cols[idx])
            new_cols.append(f_cols[idx])
            new_norms.append(f_norms[idx])
            new_norms.append(f_norms[idx])
            new_ts.append(f_two_sided[idx])
            new_ts.append(f_two_sided[idx])

        # --- 2 вершины за камерой: остаётся 1 -> 1 треугольник ---
        m2 = behind == 2
        if np.any(m2):
            idx = np.nonzero(m2)[0]
            # Вершина ПЕРЕД камерой (единственная с d >= 0)
            in_front = (d0 >= 0) | (d1 >= 0) | (d2 >= 0)
            i0 = np.argmax(np.stack([d0, d1, d2]) >= 0.0, axis=0)[idx]
            va = np.stack([v0, v1, v2], axis=1)[idx, i0]
            da = np.stack([d0, d1, d2], axis=1)[idx, i0]
            i1 = (i0 + 1) % 3
            i2 = (i0 + 2) % 3
            vb = np.stack([v0, v1, v2], axis=1)[idx, i1]
            vc = np.stack([v0, v1, v2], axis=1)[idx, i2]
            db = np.stack([d0, d1, d2], axis=1)[idx, i1]
            dc = np.stack([d0, d1, d2], axis=1)[idx, i2]
            p_ab = _cut(va, vb, da, db)
            p_ac = _cut(va, vc, da, dc)
            new_v0.append(p_ab)
            new_v1.append(p_ac)
            new_v2.append(va)
            new_cols.append(f_cols[idx])
            new_norms.append(f_norms[idx])
            new_ts.append(f_two_sided[idx])

        # --- 3 вершины за камерой: треугольник не виден вовсе ---
        # (эти случаи отфильтрованы до вызова)

        if not new_v0:
            # Ни одна грань не пересекает ближнюю плоскость - возвращаем
            # пустые массивы правильной формы (иначе vstack падает).
            empty = np.zeros((0, 3), dtype=np.float32)
            return (
                empty, empty, empty,
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=bool),
            )

        return (
            np.vstack(new_v0),
            np.vstack(new_v1),
            np.vstack(new_v2),
            np.vstack(new_cols),
            np.vstack(new_norms),
            np.concatenate(new_ts),
        )

    @staticmethod
    def _make_surface(size: Tuple[int, int]) -> pygame.Surface:
        """
        Создаёт поверхность, привязанную к формату экрана, если он есть.

        `convert()` требует открытого окна; в автономных проверках
        (--self-test) окна нет, поэтому там остаётся обычная поверхность.
        """
        surf = pygame.Surface(size)
        try:
            if pygame.display.get_surface() is not None:
                surf = surf.convert()
        except pygame.error:
            pass
        return surf

    @staticmethod
    def _blit_rgb(surface: pygame.Surface, rgb: np.ndarray) -> None:
        """Быстрая заливка поверхности массивом RGB (HxWx3, uint8)."""
        arr = pygame.surfarray.pixels3d(surface)
        arr[:, :, :] = rgb
        del arr

    @staticmethod
    def _add_glint_row(
        surface: pygame.Surface,
        x0: int,
        x1: int,
        y: int,
        prof: np.ndarray,
        color: np.ndarray,
    ) -> None:
        """Добавляет мягкий аддитивный блик в одну строку поверхности."""
        img = pygame.surfarray.pixels3d(surface)
        strip = img[x0:x1, y, :].astype(np.int16)
        strip += (color[None, :] * prof[:, None]).astype(np.int16)
        np.clip(strip, 0, 255, out=strip)
        img[x0:x1, y, :] = strip.astype(np.uint8)
        del img


    def _cloud_scaled(
        self,
        color: Tuple[int, int, int],
        sw: int,
        sh: int,
        alpha: int,
        variant: int = 0,
    ) -> Optional[Tuple[pygame.Surface, int, int]]:
        """
        Кэшированный спрайт облака: нужного размера, цвета и прозрачности.

        Масштабирование + тонировка на каждый кадр для 50+ облаков были
        главным источником тормозов (24 мс -> 5 мс после кэша).
        Размер, цвет и альфа квантуются, поэтому количество уникальных
        спрайтов ограничено.

        Возвращает (спрайт, фактическая_ширина, фактическая_высота) - реальные
        размеры нужны, чтобы позиционировать облако по центру.
        """
        sw_q = max(32, min(480, ((sw + 15) // 16) * 16))
        sh_q = max(16, ((sh + 7) // 8) * 8)
        cq = (int(color[0]) // 32, int(color[1]) // 32, int(color[2]) // 32)
        aq = max(1, min(255, int(alpha)))
        variant = int(variant) % len(self._cloud_sprites)
        key = (sw_q, sh_q, cq[0], cq[1], cq[2], aq, variant)
        cached = self._cloud_cache.get(key)
        if cached is None:
            if len(self._cloud_cache) > 240:
                self._cloud_cache.clear()
            quant_col = (cq[0] * 32, cq[1] * 32, cq[2] * 32)
            tinted = self._tinted(self._cloud_sprites[variant], quant_col)
            scaled = pygame.transform.smoothscale(tinted, (sw_q, sh_q))
            # Запекаем прозрачность прямо в альфа-канал, чтобы не вызывать
            # set_alpha() на кэшированном спрайте (это ломало бы кэш).
            arr = pygame.surfarray.pixels_alpha(scaled).astype(np.float32)
            arr = arr * (aq / 255.0)
            pygame.surfarray.pixels_alpha(scaled)[:] = arr.astype(np.uint8)
            cached = scaled
            self._cloud_cache[key] = cached
        return (cached, sw_q, sh_q)

    # ------------------------------------------------------------------
    # Проекция направления на небесной сфере в экранные координаты
    # ------------------------------------------------------------------
    def _project_dir(
        self,
        az: float,
        el: float,
        rot_world_to_cam: np.ndarray,
        focal: float,
        cx: float,
        cy: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Проецирует мировое направление (азимут, высота) в экранные координаты.

        Используется полная матрица поворота камеры (включая roll), поэтому
        солнце/луна/звёзды корректно «уезжают» при наклоне FPV-камеры.
        Возвращает None, если направление за спиной камеры.
        """
        d_world = np.array(
            [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
            dtype=np.float32,
        )
        d_cam = rot_world_to_cam @ d_world
        if d_cam[2] <= 0.05:
            return None
        sx = cx + focal * (float(d_cam[0]) / float(d_cam[2]))
        sy = cy - focal * (float(d_cam[1]) / float(d_cam[2]))
        return (sx, sy)

    def _project_dirs_batch(
        self,
        az: np.ndarray,
        el: np.ndarray,
        rot_world_to_cam: np.ndarray,
        focal: float,
        cx: float,
        cy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Векторная проекция множества направлений (для звёзд и облаков).
        Возвращает (sx, sy) массивы и маску `z > 0` (направление не за спиной).
        """
        ce = np.cos(el).astype(np.float32)
        d_world = np.stack(
            [ce * np.sin(az), np.sin(el).astype(np.float32), ce * np.cos(az)],
            axis=1,
        )
        # Тот же порядок, что и в _project_dir: rot_world_to_cam @ d_world
        d_cam = (rot_world_to_cam @ d_world.T).T
        z = d_cam[:, 2]
        safe_z = np.maximum(z, 1e-3)
        sx = cx + focal * (d_cam[:, 0] / safe_z)
        sy = cy - focal * (d_cam[:, 1] / safe_z)
        return sx, sy, z > 0.05

    @staticmethod
    def _horizon_line(cam: Camera3D, focal: float, cx: float, cy: float) -> Tuple[float, float]:
        """
        Прямая горизонта в экранных координатах: (sy = k * (sx - cx) + b).

        В пространстве камеры горизонт (мировая плоскость y = 0) задаётся
        уравнением cp*sr*x + cp*cr*y - sp*z = 0, откуда после подстановки
        x = (sx-cx)*z/focal, y = -(sy-cy)*z/focal получаем наклон и смещение.
        """
        m = euler_to_matrix(cam.yaw, cam.pitch, cam.roll)
        cp = math.cos(cam.pitch)
        sp = math.sin(cam.pitch)
        sr = math.sin(cam.roll)
        cr = math.cos(cam.roll)
        denom = cp * cr
        if abs(denom) < 1e-4:
            denom = 1e-4 if denom >= 0 else -1e-4
        k = sr / cr if abs(cr) > 1e-4 else 0.0
        b = cy - (sp * focal) / denom
        return (k, b)

    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    return pygame.font.Font(p, size)
                except OSError:
                    continue
        return pygame.font.Font(None, size)

    def _load_chereshnya_thumb(self) -> Optional[pygame.Surface]:
        img_path = os.path.join(self.repo_root, "черешня.jpg")
        if os.path.isfile(img_path):
            try:
                raw = pygame.image.load(img_path)
                return pygame.transform.smoothscale(raw, (68, 58))
            except pygame.error:
                return None
        return None

    def resize(self, width: int, height: int) -> None:
        self.width = max(320, width)
        self.height = max(240, height)
        self._world_buf = None
        self._world_up = None

    def set_render_scale(self, scale: float) -> None:
        """
        Задаёт желаемый масштаб внутреннего разрешения мира (0.5..1.0).

        1.0 - рисовать в натуральном разрешении; меньше - быстрее, но
        картинка мягче. HUD и прицел рисуются поверх в полном разрешении.
        """
        self._render_scale_target = max(0.5, min(1.0, float(scale)))

    def _update_render_scale(self, frame_ms: float) -> None:
        """
        Подстраивает масштаб под реальную производительность.

        Если кадр стабильно укладывается в бюджет - возвращаем полное
        разрешение, если нет - постепенно снижаем масштаб (не чаще раза
        в 30 кадров, чтобы не было «дыхания» картинки).
        """
        a = self._frame_ms_avg
        self._frame_ms_avg = frame_ms if a <= 0.0 else a * 0.9 + frame_ms * 0.1
        if self._scale_cooldown > 0.0:
            self._scale_cooldown -= 1.0
            return
        target = self._render_scale_target
        avg = self._frame_ms_avg
        if avg > 26.0 and self.render_scale > 0.5:
            # Не укладываемся - уменьшаем разрешение мира
            self.render_scale = max(0.5, self.render_scale - 0.1)
            self._scale_cooldown = 30
        elif avg < 15.0 and self.render_scale < target:
            # Есть запас - возвращаем качество
            self.render_scale = min(target, self.render_scale + 0.1)
            self._scale_cooldown = 45
        elif abs(self.render_scale - target) > 0.001 and self.render_scale > target:
            self.render_scale = max(target, self.render_scale - 0.1)
            self._scale_cooldown = 30

    # -----------------------------------------------------------------------
    # Camera Builders for Both Roles
    # -----------------------------------------------------------------------
    def make_drone_camera(self, sim: WorldSimulation) -> Camera3D:
        if sim.drone_alive:
            fwd = forward_from_angles(sim.drone_yaw, sim.drone_pitch)
            cam_pos = sim.drone_pos + fwd * 0.42 + np.array([0.0, 0.08, 0.0], dtype=np.float32)
            return Camera3D(
                pos=cam_pos,
                yaw=sim.drone_yaw,
                pitch=sim.drone_pitch,
                roll=sim.drone_roll,
                fov_deg=104.0,
            )
        else:
            # Spectator view watching explosion / truck while awaiting next FPV launch
            cam_pos = sim.truck_pos + np.array([-16.0, 14.0, -24.0], dtype=np.float32)
            rel = sim.truck_pos - cam_pos
            yaw = math.atan2(float(rel[0]), float(rel[2]))
            pitch = 0.35
            return Camera3D(pos=cam_pos, yaw=yaw, pitch=pitch, roll=0.0, fov_deg=80.0)

    def make_truck_camera(self, sim: WorldSimulation) -> Camera3D:
        # Gunner/Driver hybrid view anchored above & slightly behind roof AA mount
        muzzle = sim.get_turret_muzzle_pos()
        aim_fwd = forward_from_angles(sim.turret_yaw, sim.turret_pitch)
        cam_pos = muzzle - aim_fwd * 4.8 + np.array([0.0, 1.45, 0.0], dtype=np.float32)
        cam_pos[1] = max(float(cam_pos[1]), ROAD_DECK_Y + 2.2)
        return Camera3D(
            pos=cam_pos,
            yaw=sim.turret_yaw,
            pitch=sim.turret_pitch,
            roll=0.0,
            fov_deg=80.0,
        )

    # -----------------------------------------------------------------------
    # Core Vectorized 3D Projection & Painter's Rasterizer
    # -----------------------------------------------------------------------
    def render_world(
        self,
        surface: pygame.Surface,
        sim: WorldSimulation,
        cam: Camera3D,
        hide_own_drone: bool = False,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Renders full 3D scene onto `surface`.
        Returns projected screen `(sx, sy, dist)` of the opponent target if in front of camera.
        """
        w_surf, h_surf = surface.get_size()
        weather = sim.weather
        flash = sim.lightning_flash
        self._sky_time += 1.0 / 60.0

        # --- Внутренний буфер мира пониженного разрешения --------------------
        # Весь 3D-мир рисуется в буфер bw x bh, затем растягивается на
        # экран. Экранный HUD/прицел рисуется поверх в полном разрешении.
        _t_start = time.perf_counter()
        scale = self.render_scale
        if scale >= 0.999:
            canvas = surface
            upscale = False
        else:
            bw = max(320, int(w_surf * scale))
            bh = max(240, int(h_surf * scale))
            buf = self._world_buf
            if buf is None or buf.get_size() != (bw, bh):
                buf = self._make_surface((bw, bh))
                self._world_buf = buf
                self._world_up = None
            canvas = buf
            upscale = True

        w_canvas, h_canvas = canvas.get_size()

        # 1. Sky Gradient + Horizon Sea Band (with horizon roll tilt for FPV immersion!)
        self._draw_sky_and_horizon(canvas, cam, weather, flash)

        # Camera inverse rotation matrix (World -> Camera space where +Z is forward, +X right, +Y up)
        rot_wc = euler_to_matrix(cam.yaw, cam.pitch, cam.roll).T
        fov_rad = math.radians(cam.fov_deg)
        focal = (w_canvas * 0.5) / math.tan(fov_rad * 0.5)
        cx_scr = w_canvas * 0.5
        cy_scr = h_canvas * 0.5

        # Collect world triangles from static bridge + dynamic entities
        vert_chunks: List[np.ndarray] = [self.bridge_mesh.vertices]
        face_chunks: List[np.ndarray] = [self.bridge_mesh.faces]
        col_chunks: List[np.ndarray] = [self.bridge_mesh.colors]
        norm_chunks: List[np.ndarray] = [self.bridge_mesh.normals]
        ts_chunks: List[np.ndarray] = [self.bridge_mesh.two_sided]
        v_offset = len(self.bridge_mesh.vertices)

        def append_transformed(mesh: Mesh3D, pos: np.ndarray, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
            nonlocal v_offset
            rmat = euler_to_matrix(yaw, pitch, roll)
            tv = mesh.vertices @ rmat.T + pos
            tn = mesh.normals @ rmat.T
            vert_chunks.append(tv)
            face_chunks.append(mesh.faces + v_offset)
            col_chunks.append(mesh.colors)
            norm_chunks.append(tn)
            ts_chunks.append(mesh.two_sided)
            v_offset += len(mesh.vertices)

        # Add Fuel Truck + Roof Turret
        has_cage = sim.truck_mod == "anti_drone_cage"
        twin_zu = sim.truck_mod == "twin_zu_turret"
        append_transformed(self.truck_meshes[has_cage], sim.truck_pos, sim.truck_yaw)
        append_transformed(
            self.turret_meshes[twin_zu],
            sim.get_turret_muzzle_pos() - np.array([0.0, 0.32, 0.0], dtype=np.float32),
            sim.turret_yaw,
            sim.turret_pitch,
        )

        # Add FPV Drone (if alive and not hidden by first-person cockpit)
        if sim.drone_alive and not hide_own_drone:
            shaped = sim.drone_mod == "shaped_charge"
            append_transformed(
                self.drone_meshes[shaped],
                sim.drone_pos,
                sim.drone_yaw,
                sim.drone_pitch,
                sim.drone_roll,
            )

        all_verts = np.vstack(vert_chunks)
        all_faces = np.vstack(face_chunks)
        all_cols = np.vstack(col_chunks)
        all_norms = np.vstack(norm_chunks)
        all_two_sided = np.concatenate(ts_chunks)

        # Transform vertices to camera space
        cam_verts = (all_verts - cam.pos) @ rot_wc.T  # (V, 3)

        v0 = cam_verts[all_faces[:, 0]]
        v1 = cam_verts[all_faces[:, 1]]
        v2 = cam_verts[all_faces[:, 2]]

        # --- Отсечение по ближней и дальней плоскостям -------------------------
        # Раньше грани, пересекающие ближнюю плоскость, просто отбрасывались
        # целиком: вблизи камеры (например, когда дрон летит над полотном)
        # пропадали куски моста. Теперь треугольник разрезается по плоскости
        # z = near_z и получается 1 или 2 треугольника.
        near_z = 0.65
        far_z = weather.fog_end * 1.28
        min_z = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
        max_z = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2])

        # Дальняя плоскость: целиком за ней - выкидываем.
        far_mask = max_z < far_z
        # Ближняя: целиком за камерой - тоже выкидываем.
        near_out_mask = max_z <= near_z
        keep_mask = far_mask & ~near_out_mask

        if np.any(keep_mask):
            v0 = v0[keep_mask]
            v1 = v1[keep_mask]
            v2 = v2[keep_mask]
            f_cols = all_cols[keep_mask]
            f_norms = all_norms[keep_mask]
            f_two_sided = all_two_sided[keep_mask]

            # --- Разрезание треугольников по ближней плоскости ---------------
            # Треугольник может пересекать z = near_z: тогда он делится на
            # части. Обрабатываем ТОЛЬКО пересекающие грани (обычно их
            # единицы), остальные идут дальше без изменений.
            z0 = v0[:, 2]
            z1 = v1[:, 2]
            z2 = v2[:, 2]
            cross = (z0 < near_z) | (z1 < near_z) | (z2 < near_z)
            if np.any(cross):
                cv0, cv1, cv2, ccols, cnorms, cts = self._clip_near_plane(
                    v0[cross], v1[cross], v2[cross],
                    f_cols[cross], f_norms[cross], f_two_sided[cross],
                    near_z,
                )
                plain = ~cross
                v0 = np.vstack([v0[plain], cv0])
                v1 = np.vstack([v1[plain], cv1])
                v2 = np.vstack([v2[plain], cv2])
                f_cols = np.vstack([f_cols[plain], ccols])
                f_norms = np.vstack([f_norms[plain], cnorms])
                f_two_sided = np.concatenate([f_two_sided[plain], cts])

            # Perspective projection: sx = cx + (x / z) * focal, sy = cy - (y / z) * focal
            inv_z0 = focal / v0[:, 2]
            inv_z1 = focal / v1[:, 2]
            inv_z2 = focal / v2[:, 2]

            sx0 = cx_scr + v0[:, 0] * inv_z0
            sy0 = cy_scr - v0[:, 1] * inv_z0
            sx1 = cx_scr + v1[:, 0] * inv_z1
            sy1 = cy_scr - v1[:, 1] * inv_z1
            sx2 = cx_scr + v2[:, 0] * inv_z2
            sy2 = cy_scr - v2[:, 1] * inv_z2

            # 2D Screen-space bounding box + корректное отсечение граней
            #
            # ВАЖНО: раньше backface-отсечение считалось по знаку экранной площади
            # (area2 < 1e4). Это ломало две вещи:
            #   * большие фронтальные грани (полотно моста, арки, море) с площадью
            #     больше 10 000 px^2 попадали под отсечение и ИСЧЕЗАЛИ из кадра;
            #   * ~50% задних граней рисовались зря и перекрывали правильные.
            # Теперь отсечение идёт по настоящей 3D-нормали грани, а экранная
            # площадь используется только для отбрасывания суб-пиксельных граней.
            edge1 = v1 - v0
            edge2 = v2 - v0
            fn = np.cross(edge1, edge2)          # внешняя нормаль грани (обход CCW)
            fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
            fn = fn / np.maximum(fn_len, 1e-9)
            # В пространстве камеры камера находится в начале координат,
            # поэтому вектор "от грани к камере" = -центр грани.
            tri_center = (v0 + v1 + v2) * (1.0 / 3.0)
            to_cam = -tri_center
            to_cam_len = np.linalg.norm(to_cam, axis=1, keepdims=True)
            to_cam = to_cam / np.maximum(to_cam_len, 1e-9)
            facing = np.sum(fn * to_cam, axis=1)          # >0 -> грань смотрит на камеру

            area2 = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0)
            on_screen = (
                ((facing > -0.002) | f_two_sided)        # фронтальные + почти реброные
                & (np.abs(area2) > 4.0)                  # выкидываем суб-пиксельные
                & (np.maximum(np.maximum(sx0, sx1), sx2) >= -80)
                & (np.minimum(np.minimum(sx0, sx1), sx2) <= w_canvas + 80)
                & (np.maximum(np.maximum(sy0, sy1), sy2) >= -80)
                & (np.minimum(np.minimum(sy0, sy1), sy2) <= h_canvas + 80)
            )

            if np.any(on_screen):
                sx0, sy0 = sx0[on_screen], sy0[on_screen]
                sx1, sy1 = sx1[on_screen], sy1[on_screen]
                sx2, sy2 = sx2[on_screen], sy2[on_screen]
                v0_c, v1_c, v2_c = v0[on_screen], v1[on_screen], v2[on_screen]
                f_cols = f_cols[on_screen]
                f_norms = f_norms[on_screen]
                f_two_sided = f_two_sided[on_screen]
                fn = fn[on_screen]
                facing = facing[on_screen]

                # Painter's depth sort: back-to-front по САМОЙ ДАЛЬНЕЙ вершине.
                # Раньше сортировка шла по центроиду: у больших граней (полотно
                # моста, вода, арки) центроид оказывался ближе, чем стоящие на них
                # объекты, и они «проваливались» под грани - ещё одна причина
                # исчезающих кусков графики.
                d0 = np.sum(v0_c * v0_c, axis=1)
                d1 = np.sum(v1_c * v1_c, axis=1)
                d2 = np.sum(v2_c * v2_c, axis=1)
                depths = np.maximum(np.maximum(d0, d1), d2)
                order = np.argsort(depths)[::-1]

                sx0, sy0 = sx0[order].astype(np.int32), sy0[order].astype(np.int32)
                sx1, sy1 = sx1[order].astype(np.int32), sy1[order].astype(np.int32)
                sx2, sy2 = sx2[order].astype(np.int32), sy2[order].astype(np.int32)
                dist_arr = np.sqrt(depths[order])
                f_cols = f_cols[order]
                f_norms = f_norms[order]

                # Directional sun shading + weather ambient + distance fog
                #
                # Для двусторонних граней освещение считается по нормали,
                # направленной к камере: иначе «изнанка» тонкой балки была бы
                # чёрной. Остальные грани освещаются как раньше - по нормали
                # меша.
                ndotl = np.clip(f_norms @ self.sun_dir, 0.0, 1.0)
                if np.any(f_two_sided):
                    fn_use = np.where(
                        (facing < 0.0)[:, None], -fn, fn
                    )
                    ndotl = np.where(
                        f_two_sided,
                        np.maximum(ndotl, np.clip(fn_use @ self.sun_dir, 0.0, 1.0)),
                        ndotl,
                    )
                light = np.clip(weather.ambient_light * (0.62 + 0.52 * ndotl) + flash * 0.45, 0.15, 1.45)
                lit_cols = f_cols * light[:, None]

                fog_t = np.clip(
                    (dist_arr - weather.fog_start) / max(weather.fog_end - weather.fog_start, 1.0),
                    0.0,
                    1.0,
                )[:, None]
                fog_rgb = np.array(weather.fog_color, dtype=np.float32) * (1.0 + flash * 0.35)
                final_cols = np.clip(lit_cols * (1.0 - fog_t) + fog_rgb * fog_t, 0, 255).astype(np.uint8)

                # Pack coordinates & draw polygons rapidly
                pts_packed = np.stack(
                    [
                        np.stack([sx0, sy0], axis=1),
                        np.stack([sx1, sy1], axis=1),
                        np.stack([sx2, sy2], axis=1),
                    ],
                    axis=1,
                ).tolist()
                cols_list = final_cols.tolist()

                draw_poly = pygame.draw.polygon
                for pts_tri, c_tri in zip(pts_packed, cols_list):
                    draw_poly(canvas, c_tri, pts_tri)

        # 2. Draw 3D Volumetric FX: Smoke Clouds, AA Tracers, Fireball Explosions, EW Dome
        self._draw_3d_fx(canvas, sim, cam, rot_wc, focal, cx_scr, cy_scr)

        # 3. Weather Rain / Snow streaks overlay
        if weather.rain_intensity > 0.05:
            self._draw_precipitation(canvas, sim, weather)

        # Compute opponent marker screen projection
        target_world = (
            sim.truck_pos + np.array([0.0, 2.3, 0.0], dtype=np.float32)
            if hide_own_drone
            else sim.drone_pos
        )
        # --- Растягиваем буфер мира на экран ---------------------------------
        if upscale:
            up = self._world_up
            if up is None or up.get_size() != (w_surf, h_surf):
                up = self._make_surface((w_surf, h_surf))
                self._world_up = up
            pygame.transform.smoothscale(canvas, (w_surf, h_surf), up)
            surface.blit(up, (0, 0))
        self._update_render_scale((time.perf_counter() - _t_start) * 1000.0)

        tc = (target_world - cam.pos) @ rot_wc.T
        if tc[2] > 0.8:
            tsx = int(cx_scr + (tc[0] / tc[2]) * focal)
            tsy = int(cy_scr - (tc[1] / tc[2]) * focal)
            tdist = float(np.linalg.norm(tc))
            if upscale:
                # Возвращаем координаты в систему экрана (HUD рисуется
                # в полном разрешении).
                k = w_surf / float(w_canvas)
                tsx = int(tsx * k)
                tsy = int(tsy * k)
            return (tsx, tsy, tdist)
        return None

    def _draw_sky_and_horizon(
        self,
        surface: pygame.Surface,
        cam: Camera3D,
        weather: WeatherPreset,
        flash: float,
    ) -> None:
        """
        Рисует небо, светило и море.

        Ключевые особенности:
        * градиент неба считается по УГЛУ возвышения над горизонтом, а не по
          экранным полосам - поэтому закат выглядит правильно при любом наклоне
          FPV-камеры;
        * горизонт учитывает roll (наклон) камеры: линия горизонта наклоняется
          вместе с миром, а не остаётся горизонтальной;
        * море рисуется экранным проходом (без 3D-геометрии): плавный градиент
          без горизонтальных полос + солнечная дорожка + огни города;
        * небо собирается целиком в numpy и выводится одним blit - это в
          несколько раз быстрее сотен отдельных alpha-blit с облаками.
        """
        w, h = surface.get_size()
        top_c = tuple(min(255, int(c + flash * 120)) for c in weather.sky_top)
        hor_c = tuple(min(255, int(c + flash * 110)) for c in weather.sky_horizon)
        sea_c = tuple(min(255, int(c + flash * 60)) for c in weather.sea_color)

        focal = (w * 0.5) / math.tan(math.radians(cam.fov_deg) * 0.5)
        cx = w * 0.5
        cy = h * 0.5

        # Матрица "мир -> камера" (та же, что используется в 3D-проекции)
        rot_wc = euler_to_matrix(cam.yaw, cam.pitch, cam.roll).T

        # Горизонт: наклонная прямая sy = k*(sx-cx) + b
        hk, hb = self._horizon_line(cam, focal, cx, cy)

        def horizon_at(sx: float) -> float:
            return hk * (sx - cx) + hb

        # ==================================================================
        # Буфер неба в половинном разрешении
        # ==================================================================
        bw = max(2, w // 2)
        bh = max(2, h // 2)
        buf = self._sky_buf
        if buf is None or buf.get_size() != (bw, bh):
            buf = self._make_surface((bw, bh))
            self._sky_buf = buf
            self._sky_up = None
        up = self._sky_up
        if up is None or up.get_size() != (w, h):
            up = self._make_surface((w, h))
            self._sky_up = up

        b_focal = focal * 0.5
        b_cx = cx * 0.5
        b_cy = cy * 0.5

        def b_horizon_at(sx: float) -> float:
            return hk * (sx - b_cx) + hb * 0.5

        # --- Градиент неба ------------------------------------------------------
        # Цвет берём по направлению взгляда: чем выше строка над горизонтом,
        # тем ближе к зениту (top_c), у горизонта - hor_c.
        sy_h_center = b_cy - b_focal * math.tan(cam.pitch)
        rows = np.arange(bh, dtype=np.float32)
        d_center = (sy_h_center - (rows + 0.5)) / b_focal
        t_arr = np.clip(d_center / 1.05, 0.0, 1.0)
        tt_arr = t_arr * t_arr * (3.0 - 2.0 * t_arr)
        sky_rgb = np.stack(
            [
                hor_c[0] + (top_c[0] - hor_c[0]) * tt_arr,
                hor_c[1] + (top_c[1] - hor_c[1]) * tt_arr,
                hor_c[2] + (top_c[2] - hor_c[2]) * tt_arr,
            ],
            axis=1,
        ).astype(np.float32)            # (bh, 3)

        # --- Море: экранный проход ---------------------------------------------
        # Строки НИЖЕ горизонта показывают воду. Расстояние до точки на воде
        # считается по лучу взгляда: угол вниз от горизонта -> дистанция.
        sea_top = sea_c
        sea_bot = tuple(int(c * 0.55) for c in sea_c)
        fog_rgb = np.array(weather.fog_color, dtype=np.float32) * (1.0 + flash * 0.35)
        fog_start = float(weather.fog_start)
        fog_end = float(weather.fog_end)
        cam_h = max(float(cam.pos[1]), 0.5)

        below_arr = (rows + 0.5 - b_horizon_at(b_cx)) / b_focal
        below_safe = np.maximum(below_arr, 1e-3)
        dist_arr = np.minimum(cam_h / below_safe, 4000.0)
        fog_t_arr = np.clip(
            (dist_arr - fog_start) / max(fog_end - fog_start, 1.0), 0.0, 1.0
        ).astype(np.float32)
        depth_arr = np.clip(dist_arr / 900.0, 0.0, 1.0).astype(np.float32)
        sea_base = np.stack(
            [
                sea_top[0] + (sea_bot[0] - sea_top[0]) * depth_arr,
                sea_top[1] + (sea_bot[1] - sea_top[1]) * depth_arr,
                sea_top[2] + (sea_bot[2] - sea_top[2]) * depth_arr,
            ],
            axis=1,
        ).astype(np.float32)
        sea_rgb = np.clip(
            sea_base + (fog_rgb[None, :] - sea_base) * fog_t_arr[:, None], 0, 255
        ).astype(np.float32)            # (bh, 3)

        # Границы моря в каждой строке считаются аналитически (ниже),
        # поэтому общая маска bw x bh не нужна.

        # Заливка неба и моря построчно (draw.rect быстрее numpy на таких
        # объёмах: 360 строк - доли миллисекунды).
        draw_rect = pygame.draw.rect
        for y in range(bh):
            c = (
                int(sky_rgb[y, 0]),
                int(sky_rgb[y, 1]),
                int(sky_rgb[y, 2]),
            )
            draw_rect(buf, c, (0, y, bw, 1))
        sea_cols = sea_rgb.astype(np.uint8)
        # Горизонт - прямая линия, поэтому море в каждой строке занимает
        # непрерывный отрезок. Границы считаются АНАЛИТИЧЕСКИ (без маски
        # bw x bh и np.nonzero на каждую строку) - это экономит ~0.7 мс.
        for y in range(bh):
            yc = float(rows[y]) + 0.5
            if hk > 1e-6:
                # Горизонт идёт вверх вправо: море слева от линии
                x_edge = b_cx + (yc - hb * 0.5) / hk
                x_lo, x_hi = 0, int(min(bw, max(0.0, x_edge)))
            elif hk < -1e-6:
                # Горизонт идёт вниз вправо: море справа от линии
                x_edge = b_cx + (yc - hb * 0.5) / hk
                x_lo, x_hi = int(max(0.0, min(bw, x_edge))), bw
            else:
                # Горизонт горизонтален: море либо во всей строке, либо нигде
                if yc > hb * 0.5:
                    x_lo, x_hi = 0, bw
                else:
                    x_lo, x_hi = 0, 0
            if x_hi - x_lo < 1:
                continue
            c = (int(sea_cols[y, 0]), int(sea_cols[y, 1]), int(sea_cols[y, 2]))
            draw_rect(buf, c, (x_lo, y, x_hi - x_lo, 1))

        # --- Позиция светила в мире ---------------------------------------------
        sun_az = float(getattr(weather, "sun_azimuth", 0.0))
        sun_el = float(getattr(weather, "sun_elevation", 0.5))
        sun_screen = self._project_dir(sun_az, sun_el, rot_wc, focal, cx, cy)
        glow = float(getattr(weather, "sun_glow", 0.0))

        # --- Свечение светила (аддитивно, через numpy) --------------------------
        if glow > 0.01 and sun_screen is not None:
            sx_sun, sy_sun = sun_screen
            disc_col = weather.sun_color
            halo_px = int(min(w, h) * (0.26 + 0.45 * glow)) // 2
            gsz = max(4, ((halo_px * 2 + 31) // 32) * 32)
            gkey = ("glow", gsz, int(disc_col[0]), int(disc_col[1]), int(disc_col[2]))
            gsurf = self._glow_cache.get(gkey)
            if gsurf is None:
                if len(self._glow_cache) > 32:
                    self._glow_cache.clear()
                gsurf = pygame.transform.smoothscale(
                    self._premultiplied(self._glow_sprite, disc_col, glow), (gsz, gsz)
                )
                self._glow_cache[gkey] = gsurf
            gx = int(sx_sun * 0.5) - gsz // 2
            gy = int(sy_sun * 0.5) - gsz // 2
            buf.blit(
                gsurf,
                (gx, gy),
                special_flags=pygame.BLEND_RGB_ADD,
            )

        # --- Солнечная дорожка на воде ------------------------------------------
        wave = float(getattr(weather, "sea_wave", 0.35))
        if wave > 0.05 and sun_screen is not None:
            sx_sun, sy_sun = sun_screen
            sy_h_b = b_horizon_at(b_cx)
            glint = np.array(
                [
                    min(255, int(weather.sun_color[0] * 0.92)),
                    min(255, int(weather.sun_color[1] * 0.86)),
                    min(255, int(weather.sun_color[2] * 0.72)),
                ],
                dtype=np.float32,
            )
            rows_g = 64
            for i in range(rows_g):
                frac = i / rows_g
                yy = int(sy_h_b + (bh - sy_h_b) * (frac ** 1.7))
                if yy < 0 or yy >= bh:
                    continue
                spread = 5 + frac * bw * 0.16
                wob = math.sin(self._sky_time * 1.1 + i * 0.55) * spread * 0.28
                px = sx_sun * 0.5 + wob
                strength = 0.55 * wave * (1.0 - frac * 0.35)
                x0 = int(max(0, px - spread))
                x1 = int(min(bw, px + spread + 1))
                if x1 - x0 < 2:
                    continue
                xs = np.arange(x0, x1, dtype=np.float32)
                prof = np.clip(1.0 - np.abs(xs - px) / max(spread, 1.0), 0.0, 1.0)
                prof = prof * prof * (3.0 - 2.0 * prof) * strength
                if prof.max() < 0.004:
                    continue
                self._add_glint_row(buf, x0, x1, yy, prof, glint)

        # --- Облака (слой в четвертинном разрешении + апскейл) ------------------
        cloud_amount = float(getattr(weather, "cloud_amount", 0.0))
        if cloud_amount > 0.01:
            sun_rel = sun_az - cam.yaw
            c_az = self._clouds[:, 0]
            c_el = self._clouds[:, 1]
            c_size = self._clouds[:, 2]
            c_alpha = self._clouds[:, 3]
            c_phase = self._clouds[:, 4]
            drift = (self._sky_time * 0.012 + c_phase * 0.02) % math.tau
            cx_all, cy_all, c_vis = self._project_dirs_batch(c_az + drift, c_el, rot_wc, focal, cx, cy)
            # Слой облаков - в половинном разрешении (как буфер неба):
            # вчетвертинное разрешение съедало форму перистых краёв.
            cl_w = bw
            cl_h = bh
            layer = self._cloud_layer
            if layer is None or layer.get_size() != (cl_w, cl_h):
                layer = pygame.Surface((cl_w, cl_h), pygame.SRCALPHA)
                self._cloud_layer = layer
            layer.fill((0, 0, 0, 0))
            drawn = 0
            for i in range(len(c_az)):
                calpha = float(c_alpha[i])
                if calpha * cloud_amount < 0.06 or not c_vis[i]:
                    continue
                sx = float(cx_all[i])
                sy = float(cy_all[i])
                if sy > horizon_at(sx) + 26:
                    continue
                # Экранный размер из УГЛОВОГО: ширина = focal * угол.
                # Буфер неба вдвое меньше экрана -> делим ещё на 2.
                sw = max(10, int(focal * float(c_size[i]) * 0.5))
                sh = max(5, int(sw * 0.42))
                if sw < 6 or sx < -sw * 4 or sx > w + sw * 4:
                    continue
                # Подсветка облака солнцем: сравниваем АБСОЛЮТНЫЕ азимуты
                # (раньше тут смешивались относительный и абсолютный угол,
                # поэтому облака освещались случайным образом).
                lit = 0.55 + 0.45 * max(0.0, math.cos(sun_az - float(c_az[i])))
                shade = int(clamp(calpha * cloud_amount * 235 * (0.45 + 0.55 * lit), 0, 255))
                col = (
                    int(clamp(weather.cloud_color[0] * (0.62 + 0.48 * lit), 0, 255)),
                    int(clamp(weather.cloud_color[1] * (0.62 + 0.48 * lit), 0, 255)),
                    int(clamp(weather.cloud_color[2] * (0.62 + 0.48 * lit), 0, 255)),
                )
                spr = self._cloud_scaled(col, sw, sh, shade, int(c_phase[i] * 3.0))
                if spr is None:
                    continue
                spr_surf, spr_w, spr_h = spr
                layer.blit(
                    spr_surf,
                    (int(sx * 0.5 - spr_w * 0.5), int(sy * 0.5 - spr_h * 0.5)),
                )
                drawn += 1
            if drawn:
                buf.blit(layer, (0, 0))

        # --- Вывод неба на экран -------------------------------------------------
        pygame.transform.smoothscale(buf, (w, h), up)
        surface.blit(up, (0, 0))

        # --- Звёзды (полное разрешение - чтобы были резкими) ---------------------
        star_amount = float(getattr(weather, "star_density", 0.0))
        if star_amount > 0.01:
            t = self._sky_time
            s_az = self._stars[:, 0]
            s_el = self._stars[:, 1]
            s_br = self._stars[:, 2]
            s_sz = self._stars[:, 3]
            sx_all, sy_all, s_vis = self._project_dirs_batch(s_az, s_el, rot_wc, focal, cx, cy)
            fade = np.clip(1.0 - s_el / 1.1, 0.0, 1.0) ** 0.55
            tw = 0.62 + 0.38 * np.sin(t * 2.4 + s_az * 9.0 + s_el * 5.0)
            vals = 255.0 * s_br * fade * tw * min(1.0, star_amount * 1.25)
            ok = (
                s_vis
                & (s_br <= star_amount)
                & (s_el >= 0.02)
                & (sx_all >= -4)
                & (sx_all <= w + 4)
                & (sy_all >= -4)
                & (sy_all <= h + 4)
                & (vals >= 20)
            )
            for i in np.nonzero(ok)[0]:
                v = int(clamp(vals[i], 0, 255))
                col = (v, v, min(255, v + 30))
                ix, iy = int(sx_all[i]), int(sy_all[i])
                if s_sz[i] > 1.6:
                    pygame.draw.circle(surface, col, (ix, iy), 1)
                    if s_br[i] > star_amount * 0.85:
                        pygame.draw.circle(surface, (v // 2, v // 2, v // 2), (ix, iy), 2, 1)
                else:
                    surface.fill(col, (ix, iy, 1, 1))

        # --- Диск светила (поверх, в полном разрешении) -------------------------
        # Рисуется аддитивно: иначе диск (цвет светила) оказывался темнее
        # ореола вокруг и солнце выглядело «пончиком».
        if glow > 0.01 and sun_screen is not None:
            sx, sy = sun_screen
            disc_r = max(6, int(min(w, h) * 0.030))
            dsz = max(16, ((disc_r * 2 + 15) // 16) * 16)
            dkey = ("disc", dsz, int(weather.sun_color[0]),
                    int(weather.sun_color[1]), int(weather.sun_color[2]))
            dsurf = self._glow_cache.get(dkey)
            if dsurf is None:
                if len(self._glow_cache) > 32:
                    self._glow_cache.clear()
                dsurf = pygame.transform.smoothscale(
                    self._premultiplied(self._sun_sprite, weather.sun_color, glow),
                    (dsz, dsz),
                )
                self._glow_cache[dkey] = dsurf
            surface.blit(
                dsurf,
                (int(sx - dsz * 0.5), int(sy - dsz * 0.5)),
                special_flags=pygame.BLEND_RGB_ADD,
            )

        # --- Ночные огни города на противоположном берегу ------------------------
        # Огоньки сидят на линии горизонта (берег) и слегка поднимаются вверх.
        city = float(getattr(weather, "city_lights", 0.0))
        if city > 0.01:
            city_az = 2.55
            base_p = self._project_dir(city_az, 0.0, rot_wc, focal, cx, cy)
            if base_p is not None:
                base_x, base_y = base_p
                for lx, ly, lb in self._city_lights:
                    jx = base_x + (float(lx) - 0.5) * w * 0.26
                    jy = base_y - float(ly) * max(2.0, h * 0.010)
                    if not (-10 <= jx <= w + 10 and -6 <= jy <= h):
                        continue
                    v = int(clamp(255 * lb * city, 0, 255))
                    if v < 25:
                        continue
                    surface.fill((v, int(v * 0.82), int(v * 0.45)), (int(jx), int(jy), 2, 1))

    def _draw_3d_fx(
        self,
        surface: pygame.Surface,
        sim: WorldSimulation,
        cam: Camera3D,
        rot_wc: np.ndarray,
        focal: float,
        cx_scr: float,
        cy_scr: float,
    ) -> None:
        def proj(pt: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
            v = (np.array(pt, dtype=np.float32) - cam.pos) @ rot_wc.T
            if v[2] <= 0.6:
                return None
            return (int(cx_scr + (v[0] / v[2]) * focal), int(cy_scr - (v[1] / v[2]) * focal), float(v[2]))

        # EW Jammer translucent pulse ring around truck when active
        if sim.ew_active_timer > 0.0:
            p_ew = proj((float(sim.truck_pos[0]), float(sim.truck_pos[1] + 2.8), float(sim.truck_pos[2])))
            if p_ew is not None:
                pulse_r = int(clamp((28.0 / p_ew[2]) * focal * (0.6 + 0.4 * math.sin(sim.sim_time * 12.0)), 8, 280))
                pygame.draw.circle(surface, (80, 215, 255), (p_ew[0], p_ew[1]), pulse_r, 2)

        # Bullet / Pellet Tracers
        for tr in sim.tracers:
            p0 = proj(tr.start)
            p1 = proj(tr.end)
            if p0 is not None and p1 is not None:
                pygame.draw.line(surface, tr.color, (p0[0], p0[1]), (p1[0], p1[1]), 2)

        # Aerosol Smoke Clouds
        for sm in sim.smokes:
            ps = proj(sm.pos)
            if ps is not None:
                rad_px = int(clamp((sm.radius / ps[2]) * focal, 12, 420))
                smoke_surf = pygame.Surface((rad_px * 2, rad_px * 2), pygame.SRCALPHA)
                alpha = int(clamp(sm.ttl * 48.0, 40, 195))
                pygame.draw.circle(smoke_surf, (205, 212, 218, alpha), (rad_px, rad_px), rad_px)
                pygame.draw.circle(smoke_surf, (175, 182, 190, int(alpha * 0.8)), (int(rad_px * 0.75), int(rad_px * 0.9)), int(rad_px * 0.65))
                surface.blit(smoke_surf, (ps[0] - rad_px, ps[1] - rad_px))

        # Explosions
        for ex in sim.explosions:
            pe = proj(ex.pos)
            if pe is not None:
                prog = 1.0 - (ex.ttl / max(ex.max_ttl, 0.01))
                cur_r = ex.radius * (0.35 + 0.75 * prog)
                rad_px = int(clamp((cur_r / pe[2]) * focal, 8, 380))
                pygame.draw.circle(surface, (255, 120, 25), (pe[0], pe[1]), rad_px)
                pygame.draw.circle(surface, (255, 232, 110), (pe[0], pe[1]), max(4, int(rad_px * 0.58)))

    def _draw_precipitation(self, surface: pygame.Surface, sim: WorldSimulation, weather: WeatherPreset) -> None:
        w, h = surface.get_size()
        count = int(95 * weather.rain_intensity)
        is_snow = weather.id == "winter_gale"
        col = (225, 235, 248) if is_snow else (165, 190, 220)
        phase = int(sim.sim_time * 520)
        for i in range(count):
            rx = (i * 173 + phase * (1 if is_snow else 2)) % w
            ry = (i * 293 + phase * (2 if is_snow else 5)) % h
            if is_snow:
                pygame.draw.circle(surface, col, (rx, ry), 2)
            else:
                pygame.draw.line(surface, col, (rx, ry), (rx - 5, ry + 16), 1)

    # -----------------------------------------------------------------------
    # Role-Specific Tactical HUD Overlays
    # -----------------------------------------------------------------------
    def draw_drone_hud(
        self,
        surface: pygame.Surface,
        sim: WorldSimulation,
        target_proj: Optional[Tuple[int, int, float]],
    ) -> None:
        w, h = surface.get_size()
        cx, cy = w // 2, h // 2

        # 1. EW Jamming Analog Video Snow / Scanlines when inside Truck's РЭБ field!
        jam = sim.drone_jam_level
        if jam > 0.04:
            num_bars = int(jam * 28)
            for _ in range(num_bars):
                by = int(self._rng.integers(0, h))
                bh = int(self._rng.integers(2, max(4, int(18 * jam))))
                snow_s = pygame.Surface((w, bh), pygame.SRCALPHA)
                gray = int(self._rng.integers(150, 255))
                snow_s.fill((gray, gray, gray, int(145 * jam)))
                surface.blit(snow_s, (0, by))
            if jam > 0.35:
                warn = self.font_md.render("[!] ВНИМАНИЕ: ПОДАВЛЕНИЕ РЭБ ПРОТИВНИКА! [!]", True, (255, 72, 72))
                surface.blit(warn, (cx - warn.get_width() // 2, 78))

        # 2. FPV OSD Crosshair & Artificial Horizon Ladder
        osd_green = (120, 255, 140)
        pygame.draw.circle(surface, osd_green, (cx, cy), 6, 1)
        pygame.draw.line(surface, osd_green, (cx - 24, cy), (cx - 9, cy), 2)
        pygame.draw.line(surface, osd_green, (cx + 9, cy), (cx + 24, cy), 2)

        # Horizon bar tilted by drone roll
        ca, sa = math.cos(-sim.drone_roll), math.sin(-sim.drone_roll)
        pitch_off = int(-sim.drone_pitch * 120)
        hx0, hy0 = int(cx - 75 * ca), int(cy + pitch_off - 75 * sa)
        hx1, hy1 = int(cx + 75 * ca), int(cy + pitch_off + 75 * sa)
        pygame.draw.line(surface, (140, 240, 160), (hx0, hy0), (hx1, hy1), 1)

        # 3. Target Box / Thermal Indicator
        has_thermal = sim.drone_mod == "airburst_frag"
        dist_m = float(np.linalg.norm(sim.truck_pos - sim.drone_pos))
        # Hide marker inside smoke unless operator picked Thermal/Airburst modifier!
        in_smoke = any(
            float(np.linalg.norm(sim.truck_pos - np.array(sm.pos, dtype=np.float32))) < sm.radius * 1.15
            for sm in sim.smokes
        )
        if target_proj is not None and (has_thermal or (not in_smoke and jam < 0.72)):
            tx, ty, _ = target_proj
            if 0 <= tx <= w and 0 <= ty <= h:
                box_col = (255, 220, 55) if has_thermal else (255, 85, 65)
                bsz = int(clamp(4200.0 / max(dist_m, 18.0), 12, 58))
                pygame.draw.rect(surface, box_col, (tx - bsz, ty - bsz, bsz * 2, bsz * 2), 2)
                lbl = self.font_xs.render(f"ЦЕЛЬ {int(dist_m)}м", True, box_col)
                surface.blit(lbl, (tx - lbl.get_width() // 2, ty - bsz - 18))
        else:
            # Off-screen compass arrow at top
            desired_yaw = math.atan2(
                float(sim.truck_pos[0] - sim.drone_pos[0]),
                float(sim.truck_pos[2] - sim.drone_pos[2]),
            )
            rel_ang = (desired_yaw - sim.drone_yaw + math.pi) % (2 * math.pi) - math.pi
            arrow_x = int(clamp(cx + (rel_ang / 1.2) * (w * 0.38), 40, w - 40))
            lbl = self.font_sm.render(f"v БЕНЗОВОЗ ({int(dist_m)}м) v", True, (255, 140, 60))
            surface.blit(lbl, (arrow_x - lbl.get_width() // 2, 52))

        # 4. Left & Right FPV Telemetry (Speed, Altitude, Battery, Lives, Active Perk)
        spd_kmh = int(float(np.linalg.norm(sim.drone_vel)) * 3.6)
        alt_m = max(0, int(float(sim.drone_pos[1]) - ROAD_DECK_Y))
        bat_pct = int(clamp((sim.drone_battery / max(sim.drone_max_battery, 1.0)) * 100.0, 0, 100))
        bat_col = osd_green if bat_pct > 30 else (255, 75, 65)

        panel = pygame.Surface((330, 178), pygame.SRCALPHA)
        panel.fill((10, 16, 22, 175))
        surface.blit(panel, (16, h - 194))

        mode_txt = "АКРО (ручное удержание)" if getattr(sim, "drone_acro", False) else "СТАБИЛИЗАЦИЯ (самоуровень)"
        mode_col = (255, 165, 60) if getattr(sim, "drone_acro", False) else (120, 220, 255)
        warhead_cd = float(getattr(sim, "drone_detonate_cd", 0.0))
        if warhead_cd > 0.01:
            war_txt = f"БЧ: ПЕРЕЗАРЯДКА {warhead_cd:.1f}с"
            war_col = (255, 130, 90)
        else:
            war_txt = "БЧ: ГОТОВА К ПОДРЫВУ (ПКМ)"
            war_col = (120, 255, 170)
        lines = [
            (f"СКОРОСТЬ: {spd_kmh} км/ч | ВЫС: {alt_m} м", osd_green),
            (f"БАТАРЕЯ 6S: {bat_pct}%  ({sim.drone_battery:.0f}с)", bat_col),
            (f"ЦЕЛОСТНОСТЬ РАМЫ: {int(sim.drone_hp)}/{int(sim.drone_max_hp)}", (190, 235, 255)),
            (f"ЗАПАС БОРТОВ FPV: {'[+] ' * sim.drone_lives_left}", (255, 205, 80)),
            (f"МОД: {DRONE_MOD_BY_ID[sim.drone_mod].name_ru}", (175, 215, 255)),
            (f"РЕЖИМ: {mode_txt}", mode_col),
            (war_txt, war_col),
        ]
        for i, (txt, col) in enumerate(lines):
            surface.blit(self.font_xs.render(txt, True, col), (26, h - 186 + i * 24))

        if not sim.drone_alive and sim.drone_lives_left > 0:
            msg = self.font_lg.render(f"ЗАПУСК НОВОГО FPV ЧЕРЕЗ {sim.drone_respawn_timer:.1f}с...", True, (255, 200, 65))
            surface.blit(msg, (cx - msg.get_width() // 2, cy + 45))

        self._draw_shared_top_bar(surface, sim, role_label="ОПЕРАТОР FPV ДРОНА")

    def draw_truck_hud(
        self,
        surface: pygame.Surface,
        sim: WorldSimulation,
        drone_proj: Optional[Tuple[int, int, float]],
    ) -> None:
        w, h = surface.get_size()
        cx, cy = w // 2, h // 2

        # 1. Anti-Aircraft Gunner Reticle (shows shotgun spread circle + turret center pip)
        ret_col = (255, 90, 70) if sim.turret_overheated else (255, 228, 95)
        pygame.draw.circle(surface, ret_col, (cx, cy), 34, 1)
        pygame.draw.circle(surface, ret_col, (cx, cy), 4, 0)
        pygame.draw.line(surface, ret_col, (cx - 46, cy), (cx - 22, cy), 2)
        pygame.draw.line(surface, ret_col, (cx + 22, cy), (cx + 46, cy), 2)
        pygame.draw.line(surface, ret_col, (cx, cy - 46), (cx, cy - 22), 2)
        pygame.draw.line(surface, ret_col, (cx, cy + 22), (cx, cy + 46), 2)

        # 1b. Target lock ring: fills up while the gunner keeps the FPV inside the reticle
        lock = float(getattr(sim, "turret_lock", 0.0))
        if lock > 0.02:
            lock_col = (255, 70, 60) if lock > 0.85 else (255, 190, 70)
            pygame.draw.arc(
                surface,
                lock_col,
                (cx - 48, cy - 48, 96, 96),
                math.pi * 0.5,
                math.pi * 0.5 + math.pi * 2.0 * lock,
                3,
            )
            if lock > 0.85:
                lk = self.font_xs.render("ЗАХВАТ ЦЕЛИ!", True, (255, 80, 70))
                surface.blit(lk, (cx - lk.get_width() // 2, cy - 70))

        # 2. Drone Threat Indicator / Lead Radar (enhanced if Buckshot+Radar perk picked!)
        has_radar = sim.truck_mod == "buckshot_radar"
        if sim.drone_alive:
            dist_d = float(np.linalg.norm(sim.drone_pos - sim.truck_pos))
            if drone_proj is not None and 0 <= drone_proj[0] <= w and 0 <= drone_proj[1] <= h:
                dx_s, dy_s, _ = drone_proj
                if has_radar or dist_d < 165.0:
                    r_col = (255, 65, 65) if dist_d < 75.0 else (255, 190, 60)
                    pygame.draw.circle(surface, r_col, (dx_s, dy_s), 18, 2)
                    tag = f"FPV {int(dist_d)}м" if has_radar or dist_d < 110.0 else "УГРОЗА!"
                    lbl = self.font_xs.render(tag, True, r_col)
                    surface.blit(lbl, (dx_s - lbl.get_width() // 2, dy_s - 30))
            else:
                # Directional threat warning arrow
                desired_yaw = math.atan2(
                    float(sim.drone_pos[0] - sim.truck_pos[0]),
                    float(sim.drone_pos[2] - sim.truck_pos[2]),
                )
                rel = (desired_yaw - sim.turret_yaw + math.pi) % (2 * math.pi) - math.pi
                side_txt = "<< FPV СЛЕВА" if rel < 0 else "FPV СПРАВА >>"
                warn = self.font_sm.render(f"[!] {side_txt} ({int(dist_d)}м)", True, (255, 95, 75))
                surface.blit(warn, (cx - warn.get_width() // 2, 54))

        # 3. Left Panel: Truck Cistern HP, Speed, Bridge Progress
        panel_l = pygame.Surface((336, 134), pygame.SRCALPHA)
        panel_l.fill((12, 16, 24, 180))
        surface.blit(panel_l, (16, h - 150))

        hp_ratio = clamp(sim.truck_hp / max(sim.truck_max_hp, 1.0), 0.0, 1.0)
        hp_col = (75, 230, 115) if hp_ratio > 0.45 else (255, 80, 65)
        spd_kmh = int(sim.truck_speed * 3.6)
        prog_pct = int(
            clamp(
                (float(sim.truck_pos[2]) - BRIDGE_START_Z) / (BRIDGE_FINISH_Z - BRIDGE_START_Z) * 100.0,
                0.0,
                100.0,
            )
        )

        # Easter-egg cabin sticker from `черешня.jpg` in corner of dashboard!
        if self.easter_egg_thumb is not None:
            surface.blit(self.easter_egg_thumb, (278, h - 142))

        l_lines = [
            (f"БРОНЯ ЦИСТЕРНЫ: {int(sim.truck_hp)}/{int(sim.truck_max_hp)} HP", hp_col),
            (f"СКОРОСТЬ ТЯГАЧА: {spd_kmh} км/ч", (235, 240, 248)),
            (f"МАРШРУТ ПО МОСТУ: {prog_pct}% до КПП", (110, 215, 255)),
            (f"ДРОНОВ У ВРАГА: {sim.drone_lives_left}", (255, 195, 90)),
            (f"МОД: {TRUCK_MOD_BY_ID[sim.truck_mod].name_ru}", (185, 215, 255)),
        ]
        for i, (txt, col) in enumerate(l_lines):
            surface.blit(self.font_xs.render(txt, True, col), (26, h - 142 + i * 24))

        # 4. Right Panel: Anti-FPV Weapons & Countermeasures Status
        panel_r = pygame.Surface((295, 134), pygame.SRCALPHA)
        panel_r.fill((12, 16, 24, 180))
        surface.blit(panel_r, (w - 311, h - 150))

        tur_txt = "ПЕРЕГРЕВ!" if sim.turret_overheated else f"Нагрев {int(sim.turret_heat * 100)}%"
        sg_txt = "ГОТОВ [ПКМ / F / RB]" if sim.shotgun_cd <= 0.01 else f"Перезарядка {sim.shotgun_cd:.1f}с"
        ew_txt = (
            f"АКТИВЕН ({sim.ew_active_timer:.1f}с)!"
            if sim.ew_active_timer > 0
            else ("ГОТОВ [E / X]" if sim.ew_cd <= 0.01 else f"Откат {sim.ew_cd:.1f}с")
        )
        sm_txt = "ГОТОВ [Q / Y]" if sim.smoke_cd <= 0.01 else f"Откат {sim.smoke_cd:.1f}с"

        r_lines = [
            (f"[ЛКМ] ЗПУ Турель: {tur_txt}", (255, 100, 80) if sim.turret_overheated else (245, 225, 110)),
            (f"[ПКМ] Дробовик: {sg_txt}", (135, 245, 155) if sim.shotgun_cd <= 0.01 else (190, 195, 205)),
            (f"[E] Купол РЭБ: {ew_txt}", (95, 225, 255)),
            (f"[Q] Дымзавеса: {sm_txt}", (215, 222, 230)),
            ("Мышь/Правый стик — прицел ПВО", (160, 175, 195)),
        ]
        for i, (txt, col) in enumerate(r_lines):
            surface.blit(self.font_xs.render(txt, True, col), (w - 301, h - 142 + i * 24))

        self._draw_shared_top_bar(surface, sim, role_label="ВОДИТЕЛЬ-СТРЕЛОК БЕНЗОВОЗА")

    def _draw_shared_top_bar(self, surface: pygame.Surface, sim: WorldSimulation, role_label: str) -> None:
        w, _ = surface.get_size()
        bar = pygame.Surface((w, 42), pygame.SRCALPHA)
        bar.fill((8, 12, 20, 195))
        surface.blit(bar, (0, 0))

        font_bar = self.font_xs if w < 1050 else self.font_sm
        short_role = "FPV ДРОН" if "FPV" in role_label else "БЕНЗОВОЗ"
        left_txt = (
            f"Р{sim.round_number} | FPV {sim.drone_score}:{sim.truck_score} БЕНЗ | {short_role}"
            if w < 1050
            else f"РАУНД {sim.round_number}  |  СЧЁТ  FPV {sim.drone_score} : {sim.truck_score} БЕНЗОВОЗ  |  {role_label}"
        )
        left = font_bar.render(left_txt, True, (240, 244, 250))
        surface.blit(left, (12, 11))

        right_str = (
            f"{sim.weather.name_ru} | {max(0, int(sim.phase_timer))}с"
            if w < 1050
            else f"Погода: {sim.weather.name_ru}  |  Осталось: {max(0, int(sim.phase_timer))}с"
        )
        right = font_bar.render(right_str, True, (135, 215, 255))
        surface.blit(right, (w - right.get_width() - 12, 11))

        # Event ticker just below top bar
        if sim.last_event_ru:
            ev = self.font_xs.render(sim.last_event_ru, True, (255, 225, 130))
            surface.blit(ev, (12, 46))

    # -----------------------------------------------------------------------
    # Pre-Round Modifier Draft & Round-End Overlays
    # -----------------------------------------------------------------------
    def draw_phase_overlay(
        self,
        surface: pygame.Surface,
        sim: WorldSimulation,
        my_role: str,
        net_status: str = "",
    ) -> None:
        w, h = surface.get_size()
        cx, cy = w // 2, h // 2

        if sim.phase == "MODIFIER_DRAFT":
            dim = pygame.Surface((w, h), pygame.SRCALPHA)
            dim.fill((8, 12, 22, 205))
            surface.blit(dim, (0, 0))

            title = self.font_lg.render(
                f"РАУНД {sim.round_number} — СВОДКА ПОГОДЫ И ВЫБОР МОДИФИКАТОРА",
                True,
                (255, 218, 95),
            )
            surface.blit(title, (cx - title.get_width() // 2, 52))

            w_box = self.font_md.render(f"Погода раунда: {sim.weather.name_ru}", True, (115, 215, 255))
            w_desc = self.font_sm.render(sim.weather.description_ru, True, (210, 220, 235))
            surface.blit(w_box, (cx - w_box.get_width() // 2, 98))
            surface.blit(w_desc, (cx - w_desc.get_width() // 2, 128))

            from .config import DRONE_MODIFIERS, TRUCK_MODIFIERS
            mods = DRONE_MODIFIERS if my_role == "drone" else TRUCK_MODIFIERS
            selected_id = sim.drone_mod if my_role == "drone" else sim.truck_mod
            am_ready = sim.drone_draft_ready if my_role == "drone" else sim.truck_draft_ready

            role_hdr = (
                "Выберите оснащение FPV Дрона (клавиши 1–5 или крестовина геймпада):"
                if my_role == "drone"
                else "Выберите модификацию Бензовоза и ПВО (клавиши 1–5 или крестовина геймпада):"
            )
            hdr_s = self.font_sm.render(role_hdr, True, (240, 245, 250))
            surface.blit(hdr_s, (cx - hdr_s.get_width() // 2, 168))

            # Карточка шире и выше, чем раньше: длинные названия и описания
            # модификаторов больше не вылезают за её края.
            card_w = min(900, w - 60)
            card_x = cx - card_w // 2
            card_h = 70
            card_step = 78
            for idx, m in enumerate(mods):
                cy_card = 196 + idx * card_step
                is_sel = m.id == selected_id
                bg = (36, 78, 122, 235) if is_sel else (22, 30, 44, 215)
                border = (255, 212, 75) if is_sel else (70, 88, 112)

                card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                card.fill(bg)
                surface.blit(card, (card_x, cy_card))
                pygame.draw.rect(surface, border, (card_x, cy_card, card_w, card_h), 2)

                # Строка 1: имя модификатора (крупный шрифт).
                t1 = self.font_md.render(f"[{m.key_hint}] {m.name_ru}", True, (255, 255, 255))
                surface.blit(t1, (card_x + 16, cy_card + 4))
                # Строка 2: краткий эффект (справа от имени не влезает -
                # поэтому эффект идёт отдельной строкой слева).
                t_short = self.font_xs.render(m.short_ru, True, (255, 210, 110))
                surface.blit(t_short, (card_x + 16, cy_card + 33))
                # Строка 3: полное описание.
                t2 = self.font_xs.render(m.desc_ru, True, (195, 210, 228))
                surface.blit(t2, (card_x + 16, cy_card + 50))

            # --- Модификатор противника (известен сразу - играем встречно) ---
            if my_role == "drone":
                opp_mod = TRUCK_MOD_BY_ID.get(sim.truck_mod)
                opp_role_txt = "Бензовоз выбрал:"
            else:
                opp_mod = DRONE_MOD_BY_ID.get(sim.drone_mod)
                opp_role_txt = "Оператор FPV выбрал:"
            opp_box_w = min(900, w - 60)
            opp_y = 596
            opp_panel = pygame.Surface((opp_box_w, 34), pygame.SRCALPHA)
            opp_panel.fill((16, 24, 36, 215))
            surface.blit(opp_panel, (cx - opp_box_w // 2, opp_y))
            pygame.draw.rect(surface, (86, 118, 150), (cx - opp_box_w // 2, opp_y, opp_box_w, 34), 1)
            opp_txt = f"{opp_role_txt} {opp_mod.name_ru if opp_mod else '—'}"
            opp_s = self.font_sm.render(opp_txt, True, (255, 175, 110))
            surface.blit(opp_s, (cx - opp_s.get_width() // 2, opp_y + 9))

            rdy_txt = (
                "[OK] ВЫ ГОТОВЫ! Ожидание второго игрока..."
                if am_ready
                else f"Нажмите [ENTER / R / Кнопка A] для готовности (автостарт: {int(sim.phase_timer)}с)"
            )
            rdy_col = (95, 245, 130) if am_ready else (255, 210, 90)
            rdy_s = self.font_md.render(rdy_txt, True, rdy_col)
            surface.blit(rdy_s, (cx - rdy_s.get_width() // 2, 640))

            if net_status:
                ns = self.font_sm.render(net_status, True, (155, 205, 255))
                surface.blit(ns, (cx - ns.get_width() // 2, 674))

        elif sim.phase == "ROUND_END":
            banner = pygame.Surface((min(780, w - 40), 170), pygame.SRCALPHA)
            banner.fill((12, 18, 28, 225))
            bx, by = cx - banner.get_width() // 2, cy - 85
            surface.blit(banner, (bx, by))
            win_col = (255, 115, 65) if sim.round_winner == "drone" else (85, 225, 130)
            pygame.draw.rect(surface, win_col, (bx, by, banner.get_width(), 170), 3)

            who = "ПОБЕДА ОПЕРАТОРА FPV ДРОНА!" if sim.round_winner == "drone" else "БЕНЗОВОЗ ВЫСТОЯЛ В РАУНДЕ!"
            t1 = self.font_lg.render(who, True, win_col)
            t2 = self.font_md.render(sim.round_reason_ru, True, (240, 244, 250))
            t3 = self.font_sm.render(f"Следующий раунд и смена погоды через {sim.phase_timer:.1f}с...", True, (180, 195, 215))
            surface.blit(t1, (cx - t1.get_width() // 2, by + 24))
            surface.blit(t2, (cx - t2.get_width() // 2, by + 76))
            surface.blit(t3, (cx - t3.get_width() // 2, by + 124))

        elif sim.phase == "MATCH_OVER":
            dim = pygame.Surface((w, h), pygame.SRCALPHA)
            dim.fill((8, 12, 20, 225))
            surface.blit(dim, (0, 0))

            win_col = (255, 125, 65) if sim.match_winner == "drone" else (85, 235, 135)
            who = "МАТЧ ЗАВЕРШЁН: ПОБЕДА ОПЕРАТОРА FPV!" if sim.match_winner == "drone" else "МАТЧ ЗАВЕРШЁН: ПОБЕДА ВОДИТЕЛЯ БЕНЗОВОЗА!"
            t1 = self.font_lg.render(who, True, win_col)
            t2 = self.font_lg.render(f"Итоговый счёт:  FPV {sim.drone_score}  —  {sim.truck_score} БЕНЗОВОЗ", True, (245, 248, 255))
            t3 = self.font_md.render("Нажмите [ENTER / R / A] для реванша или [ESC] для выхода в главное меню", True, (200, 215, 235))
            surface.blit(t1, (cx - t1.get_width() // 2, cy - 80))
            surface.blit(t2, (cx - t2.get_width() // 2, cy - 20))
            surface.blit(t3, (cx - t3.get_width() // 2, cy + 50))
