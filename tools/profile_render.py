"""
Профилировщик рендера и проверка корректности отсечения граней.

Запуск:
    .venv/bin/python tools/profile_render.py [--shots out_dir] [--frames 120]

Что делает:
  * измеряет время кадра для каждого погодного пресета и для полного FPV-кадра;
  * проверяет, что backface-отсечение больше не выкидывает фронтальные грани
    и не оставляет задние;
  * (опционально) сохраняет PNG-скриншоты каждого пресета в --shots.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import WEATHER_PRESETS  # noqa: E402
from src.math3d import build_crimean_bridge_mesh, euler_to_matrix  # noqa: E402
from src.physics import WorldSimulation  # noqa: E402
from src.renderer import SceneRenderer3D  # noqa: E402


def build_sim() -> WorldSimulation:
    sim = WorldSimulation()
    return sim


def check_culling(sim: WorldSimulation, renderer: Renderer3D, cam) -> dict:
    """Повторяет часть конвейера рендера и считает статистику отсечения."""
    rot_wc = euler_to_matrix(cam.yaw, cam.pitch, cam.roll).T
    weather = sim.weather
    w, h = renderer.width, renderer.height
    focal = (w * 0.5) / np.tan(np.radians(cam.fov_deg) * 0.5)

    mesh = renderer.bridge_mesh
    verts = mesh.vertices
    faces = mesh.faces
    cam_verts = (verts - cam.pos) @ rot_wc.T
    v0 = cam_verts[faces[:, 0]]
    v1 = cam_verts[faces[:, 1]]
    v2 = cam_verts[faces[:, 2]]

    min_z = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
    max_z = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2])
    vis = (min_z > 0.65) & (max_z < weather.fog_end * 1.28)

    v0, v1, v2 = v0[vis], v1[vis], v2[vis]
    sx = w * 0.5 + v0[:, 0] * (focal / v0[:, 2])
    sy = h * 0.5 - v0[:, 1] * (focal / v0[:, 2])
    sx1 = w * 0.5 + v1[:, 0] * (focal / v1[:, 2])
    sy1 = h * 0.5 - v1[:, 1] * (focal / v1[:, 2])
    sx2 = w * 0.5 + v2[:, 0] * (focal / v2[:, 2])
    sy2 = h * 0.5 - v2[:, 1] * (focal / v2[:, 2])

    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    center = (v0 + v1 + v2) / 3.0
    to_cam = -center
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
    facing = np.sum(fn * to_cam, axis=1)

    area2 = (sx1 - sx) * (sy2 - sy) - (sy1 - sy) * (sx2 - sx)
    on_screen = (
        (facing > -0.002)
        & (np.abs(area2) > 4.0)
        & (np.maximum(np.maximum(sx, sx1), sx2) >= -80)
        & (np.minimum(np.minimum(sx, sx1), sx2) <= w + 80)
        & (np.maximum(np.maximum(sy, sy1), sy2) >= -80)
        & (np.minimum(np.minimum(sy, sy1), sy2) <= h + 80)
    )

    front = facing > 0.0
    back = ~front
    return {
        "visible_total": int(vis.sum()),
        "kept": int(on_screen.sum()),
        "front_kept": int((on_screen & front).sum()),
        "front_total": int(front.sum()),
        "back_kept": int((on_screen & back).sum()),
        "back_total": int(back.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    screen = pygame.display.set_mode((args.width, args.height))
    sim = build_sim()
    renderer = SceneRenderer3D(args.width, args.height, sim.barriers, repo_root)

    print(f"== Рендер-профиль {args.width}x{args.height} ==")
    for preset in WEATHER_PRESETS:
        sim.weather_id = preset.id
        cam = renderer.make_drone_camera(sim)
        # прогрев
        for _ in range(5):
            renderer.render_world(screen, sim, cam, hide_own_drone=True)
        t0 = time.perf_counter()
        for _ in range(args.frames):
            renderer.render_world(screen, sim, cam, hide_own_drone=True)
        dt = (time.perf_counter() - t0) / max(args.frames, 1) * 1000.0
        stats = check_culling(sim, renderer, cam)
        print(
            f"  {preset.id:18s} кадр={dt:6.2f} мс  fps={1000.0/max(dt,0.001):7.1f}  "
            f"грани: всего={stats['visible_total']:5d} "
            f"фронт={stats['front_total']:5d} (осталось {stats['front_kept']:5d}) "
            f"задн={stats['back_total']:5d} (осталось {stats['back_kept']:4d})"
        )
        if args.shots:
            os.makedirs(args.shots, exist_ok=True)
            pygame.image.save(screen, os.path.join(args.shots, f"weather_{preset.id}.png"))

    # Полный FPV-кадр (мир + HUD + осадки)
    sim.weather_id = WEATHER_PRESETS[0].id
    cam = renderer.make_drone_camera(sim)
    for _ in range(5):
        renderer.render_world(screen, sim, cam, hide_own_drone=True)
    t0 = time.perf_counter()
    for _ in range(args.frames):
        renderer.render_world(screen, sim, cam, hide_own_drone=True)
        renderer.draw_drone_hud(screen, sim, None)
    dt = (time.perf_counter() - t0) / max(args.frames, 1) * 1000.0
    print(f"  FPV кадр (мир+HUD)  {dt:6.2f} мс  fps={1000.0/max(dt,0.001):7.1f}")

    if args.shots:
        os.makedirs(args.shots, exist_ok=True)
        pygame.image.save(screen, os.path.join(args.shots, "fpv_frame.png"))

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
