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

        # 1. Sky Gradient + Horizon Sea Band (with horizon roll tilt for FPV immersion!)
        self._draw_sky_and_horizon(surface, cam, weather, flash)

        # Camera inverse rotation matrix (World -> Camera space where +Z is forward, +X right, +Y up)
        rot_wc = euler_to_matrix(cam.yaw, cam.pitch, cam.roll).T
        fov_rad = math.radians(cam.fov_deg)
        focal = (w_surf * 0.5) / math.tan(fov_rad * 0.5)
        cx_scr = w_surf * 0.5
        cy_scr = h_surf * 0.5

        # Collect world triangles from static bridge + dynamic entities
        vert_chunks: List[np.ndarray] = [self.bridge_mesh.vertices]
        face_chunks: List[np.ndarray] = [self.bridge_mesh.faces]
        col_chunks: List[np.ndarray] = [self.bridge_mesh.colors]
        norm_chunks: List[np.ndarray] = [self.bridge_mesh.normals]
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

        # Transform vertices to camera space
        cam_verts = (all_verts - cam.pos) @ rot_wc.T  # (V, 3)

        v0 = cam_verts[all_faces[:, 0]]
        v1 = cam_verts[all_faces[:, 1]]
        v2 = cam_verts[all_faces[:, 2]]

        # Near/Far frustum culling (fast vectorized mask)
        near_z = 0.65
        far_z = weather.fog_end * 1.28
        min_z = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
        max_z = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2])
        vis_mask = (min_z > near_z) & (max_z < far_z)

        if np.any(vis_mask):
            v0 = v0[vis_mask]
            v1 = v1[vis_mask]
            v2 = v2[vis_mask]
            f_cols = all_cols[vis_mask]
            f_norms = all_norms[vis_mask]

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

            # 2D Screen-space bounding box & backface culling
            # Signed 2D area in screen space (y-down): cross < 0 for CCW front faces
            area2 = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0)
            on_screen = (
                (area2 < 1e4)  # Keep front-facing + thin beams
                & (np.maximum(np.maximum(sx0, sx1), sx2) >= -80)
                & (np.minimum(np.minimum(sx0, sx1), sx2) <= w_surf + 80)
                & (np.maximum(np.maximum(sy0, sy1), sy2) >= -80)
                & (np.minimum(np.minimum(sy0, sy1), sy2) <= h_surf + 80)
            )

            if np.any(on_screen):
                sx0, sy0 = sx0[on_screen], sy0[on_screen]
                sx1, sy1 = sx1[on_screen], sy1[on_screen]
                sx2, sy2 = sx2[on_screen], sy2[on_screen]
                v0_c, v1_c, v2_c = v0[on_screen], v1[on_screen], v2[on_screen]
                f_cols = f_cols[on_screen]
                f_norms = f_norms[on_screen]

                # Painter's depth sort (back-to-front by average camera-space distance)
                centers_c = (v0_c + v1_c + v2_c) * (1.0 / 3.0)
                depths = np.sum(centers_c * centers_c, axis=1)
                order = np.argsort(depths)[::-1]

                sx0, sy0 = sx0[order].astype(np.int32), sy0[order].astype(np.int32)
                sx1, sy1 = sx1[order].astype(np.int32), sy1[order].astype(np.int32)
                sx2, sy2 = sx2[order].astype(np.int32), sy2[order].astype(np.int32)
                dist_arr = np.sqrt(depths[order])
                f_cols = f_cols[order]
                f_norms = f_norms[order]

                # Directional sun shading + weather ambient + distance fog
                ndotl = np.clip(f_norms @ self.sun_dir, 0.0, 1.0)
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
                    draw_poly(surface, c_tri, pts_tri)

        # 2. Draw 3D Volumetric FX: Smoke Clouds, AA Tracers, Fireball Explosions, EW Dome
        self._draw_3d_fx(surface, sim, cam, rot_wc, focal, cx_scr, cy_scr)

        # 3. Weather Rain / Snow streaks overlay
        if weather.rain_intensity > 0.05:
            self._draw_precipitation(surface, sim, weather)

        # Compute opponent marker screen projection
        target_world = (
            sim.truck_pos + np.array([0.0, 2.3, 0.0], dtype=np.float32)
            if hide_own_drone
            else sim.drone_pos
        )
        tc = (target_world - cam.pos) @ rot_wc.T
        if tc[2] > 0.8:
            tsx = int(cx_scr + (tc[0] / tc[2]) * focal)
            tsy = int(cy_scr - (tc[1] / tc[2]) * focal)
            tdist = float(np.linalg.norm(tc))
            return (tsx, tsy, tdist)
        return None

    def _draw_sky_and_horizon(
        self,
        surface: pygame.Surface,
        cam: Camera3D,
        weather: WeatherPreset,
        flash: float,
    ) -> None:
        w, h = surface.get_size()
        # Horizon Y shifts with camera pitch
        fov_v = math.radians(cam.fov_deg) * (h / max(w, 1))
        horizon_y = int(h * 0.5 - (cam.pitch / max(fov_v * 0.5, 0.1)) * (h * 0.5))
        horizon_y = int(clamp(horizon_y, -h, h * 2))

        top_c = tuple(min(255, int(c + flash * 120)) for c in weather.sky_top)
        hor_c = tuple(min(255, int(c + flash * 110)) for c in weather.sky_horizon)
        sea_c = tuple(min(255, int(c + flash * 60)) for c in weather.sea_color)

        # Banded sky gradient (12 horizontal strips = super fast & smooth)
        bands = 12
        sky_end = clamp(horizon_y, 0, h)
        if sky_end > 0:
            bh = max(1, (int(sky_end) + bands - 1) // bands)
            for i in range(bands):
                y0 = i * bh
                if y0 >= sky_end:
                    break
                t = i / max(bands - 1, 1)
                col = (
                    int(top_c[0] + (hor_c[0] - top_c[0]) * t),
                    int(top_c[1] + (hor_c[1] - top_c[1]) * t),
                    int(top_c[2] + (hor_c[2] - top_c[2]) * t),
                )
                pygame.draw.rect(surface, col, (0, y0, w, min(bh + 1, int(sky_end) - y0)))

        if sky_end < h:
            pygame.draw.rect(surface, sea_c, (0, int(sky_end), w, h - int(sky_end)))

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

            card_w = min(760, w - 80)
            card_x = cx - card_w // 2
            for idx, m in enumerate(mods):
                cy_card = 204 + idx * 68
                is_sel = m.id == selected_id
                bg = (36, 78, 122, 235) if is_sel else (22, 30, 44, 215)
                border = (255, 212, 75) if is_sel else (70, 88, 112)

                card = pygame.Surface((card_w, 60), pygame.SRCALPHA)
                card.fill(bg)
                surface.blit(card, (card_x, cy_card))
                pygame.draw.rect(surface, border, (card_x, cy_card, card_w, 60), 2)

                t1 = self.font_md.render(f"[{m.key_hint}] {m.name_ru}  ({m.short_ru})", True, (255, 255, 255))
                t2 = self.font_xs.render(m.desc_ru, True, (195, 210, 228))
                surface.blit(t1, (card_x + 16, cy_card + 7))
                surface.blit(t2, (card_x + 16, cy_card + 35))

            rdy_txt = (
                "[OK] ВЫ ГОТОВЫ! Ожидание второго игрока..."
                if am_ready
                else f"Нажмите [ENTER / R / Кнопка A] для готовности (автостарт: {int(sim.phase_timer)}с)"
            )
            rdy_col = (95, 245, 130) if am_ready else (255, 210, 90)
            rdy_s = self.font_md.render(rdy_txt, True, rdy_col)
            surface.blit(rdy_s, (cx - rdy_s.get_width() // 2, 560))

            if net_status:
                ns = self.font_sm.render(net_status, True, (155, 205, 255))
                surface.blit(ns, (cx - ns.get_width() // 2, 598))

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
