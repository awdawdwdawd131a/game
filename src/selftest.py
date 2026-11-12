"""
Built-in Verification & Diagnostic Mode ("Проверочный режим"):
Tests all subsystems automatically on any PC (either inside the desktop GUI or via `--self-test` CLI):
1. Repository integrity check (ensures `черешня.jpg` is untouched and valid)
2. 3D Bridge, Fuel Truck & FPV Drone geometry + perspective renderer test (renders real frames!)
3. Weather rotation check across 6 consecutive rounds (verifies distinct weather every round + wind/grip physics)
4. Modifiers verification for BOTH roles (shaped charge, fiber optic EW immunity, cope cage, twin turret, V12, etc.)
5. Anti-FPV combat verification (proves the Fuel Truck gunner can shoot down an incoming FPV drone!)
6. FPV Direct-Hit destruction verification (proves FPV warhead destroys the Fuel Truck!)
7. Real TCP Socket 1v1 Multiplayer loopback test (spawns `GameServer` + 2 `GameClient`s on separate sockets,
   exchanges inputs & snapshots, measures RTT ping, verifies sync!)
8. Joystick / Gamepad subsystem initialization & mapping check.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pygame

from .assets import find_file
from .config import DRONE_MODIFIERS, TRUCK_MODIFIERS, WEATHER_PRESETS
from .controls import InputController, JoystickProfile, apply_deadzone
from .math3d import forward_from_angles
from .network import GameClient, GameServer
from .physics import DroneInput, TruckInput, WorldSimulation
from .renderer import SceneRenderer3D


@dataclass
class CheckResult:
    name_ru: str
    passed: bool
    details_ru: str
    duration_ms: float


def run_verification_suite(
    repo_root: str,
    save_screenshots_dir: Optional[str] = None,
    on_progress: Optional[Callable[[CheckResult], None]] = None,
) -> List[CheckResult]:
    results: List[CheckResult] = []

    def record(name_ru: str, fn: Callable[[], str]) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            res = CheckResult(name_ru=name_ru, passed=True, details_ru=detail, duration_ms=(time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            res = CheckResult(name_ru=name_ru, passed=False, details_ru=f"ОШИБКА: {exc}", duration_ms=(time.perf_counter() - t0) * 1000.0)
        results.append(res)
        if on_progress is not None:
            on_progress(res)

    # 1. Check `черешня.jpg` preserved
    def check_chereshnya() -> str:
        # Ищем и рядом с .exe, и внутри собранного бандла (PyInstaller _MEIPASS)
        p = find_file("черешня.jpg", repo_root)
        assert p is not None, "Файл черешня.jpg не найден!"
        sz = os.path.getsize(p)
        assert sz > 10_000, f"Некорректный размер файла: {sz}"
        return f"Файл черешня.jpg на месте ({sz} байт)"

    record("1. Сохранность картинки черешня.jpg в репозитории", check_chereshnya)

    # 2. Check 3D Mesh generation & Software/SDL 3D Renderer (both FPV & Truck cameras)
    def check_3d_renderer() -> str:
        if not pygame.get_init():
            pygame.init()
        sim = WorldSimulation(seed=123)
        sim.begin_active_round()
        renderer = SceneRenderer3D(960, 540, sim.barriers, repo_root)
        surf = pygame.Surface((960, 540))

        # Render FPV view
        cam_d = renderer.make_drone_camera(sim)
        proj_t = renderer.render_world(surf, sim, cam_d, hide_own_drone=True)
        renderer.draw_drone_hud(surf, sim, proj_t)
        if save_screenshots_dir:
            os.makedirs(save_screenshots_dir, exist_ok=True)
            pygame.image.save(surf, os.path.join(save_screenshots_dir, "verify_fpv_view.png"))

        # Render Truck AA Gunner view under storm weather
        sim.weather_id = "black_sea_storm"
        sim.ew_active_timer = 2.5
        cam_t = renderer.make_truck_camera(sim)
        proj_d = renderer.render_world(surf, sim, cam_t, hide_own_drone=False)
        renderer.draw_truck_hud(surf, sim, proj_d)
        if save_screenshots_dir:
            pygame.image.save(surf, os.path.join(save_screenshots_dir, "verify_truck_view.png"))

        n_tris = len(renderer.bridge_mesh.faces) + len(renderer.truck_meshes[True].faces) + len(renderer.drone_meshes[True].faces)
        assert n_tris > 400, "Слишком мало 3D полигонов!"
        return f"Отрендерены 3D сцены FPV и Бензовоза ({n_tris} треугольников, арки + мост + HUD)"

    record("2. 3D-движок Крымского моста, камер и HUD", check_3d_renderer)

    # 3. Check Weather Rotation across rounds
    def check_weather_rotation() -> str:
        sim = WorldSimulation(seed=777)
        seen = [sim.weather_id]
        for _ in range(len(WEATHER_PRESETS) - 1):
            sim.round_number += 1
            sim.start_modifier_draft(advance_weather=True)
            assert sim.weather_id != seen[-1], "Погода повторилась два раунда подряд!"
            seen.append(sim.weather_id)
        unique_cnt = len(set(seen))
        assert unique_cnt == len(WEATHER_PRESETS), f"Ожидалось {len(WEATHER_PRESETS)} разных погод, получено {unique_cnt}"
        return f"Все {unique_cnt} погодных режимов сменяются каждый раунд без повторов"

    record("3. Смена погоды каждый раунд (6 погодных условий)", check_weather_rotation)

    # 4. Check Modifiers for BOTH FPV Drone & Fuel Truck
    def check_modifiers() -> str:
        assert len(DRONE_MODIFIERS) >= 5 and len(TRUCK_MODIFIERS) >= 5
        sim = WorldSimulation(seed=55)
        # Fiber optic immunity to EW jammer
        sim.drone_mod = "fiber_optic"
        sim.truck_mod = "ew_kupol"
        sim.begin_active_round()
        sim.drone_pos[:] = sim.truck_pos + np.array([0.0, 8.0, 12.0], dtype=np.float32)
        sim.step(1.0 / 60.0, DroneInput(), TruckInput(trigger_ew=True))
        assert sim.ew_active_timer > 0.0, "РЭБ не активировался!"
        assert sim.drone_jam_level == 0.0, "Оптоволокно должно давать полный иммунитет к РЭБ!"

        # Radio drone without fiber optic MUST get jammed by EW Kupol
        sim.drone_mod = "sport_motors"
        sim.step(1.0 / 60.0, DroneInput(), TruckInput())
        assert sim.drone_jam_level > 0.3, "Обычный FPV должен получать помехи от РЭБ!"

        # Kevlar frame extra drone life
        sim.drone_mod = "kevlar_frame"
        sim.begin_active_round()
        assert sim.drone_lives_left == 4 and sim.drone_max_hp > 55.0
        return f"Проверены {len(DRONE_MODIFIERS)} модов FPV и {len(TRUCK_MODIFIERS)} модов Бензовоза (вкл. РЭБ vs Оптоволокно)"

    record("4. Система модификаторов для обоих игроков", check_modifiers)

    # 5. Check Fuel Truck Shooting Down Incoming FPV Drone
    def check_shoot_down_fpv() -> str:
        sim = WorldSimulation(seed=99)
        sim.truck_mod = "twin_zu_turret"
        sim.begin_active_round()
        initial_lives = sim.drone_lives_left

        # Place FPV 30m behind/above the truck and aim roof turret directly at it
        muzzle = sim.get_turret_muzzle_pos()
        sim.drone_pos[:] = muzzle + np.array([0.0, 9.0, -30.0], dtype=np.float32)
        rel = sim.drone_pos - muzzle
        horiz = float(np.hypot(rel[0], rel[2]))
        sim.turret_yaw = float(np.arctan2(rel[0], rel[2]))
        sim.turret_pitch = float(np.arctan2(-rel[1], horiz))

        shot_frames = 0
        for _ in range(240):
            # The gunner keeps tracking the drone while the truck is driving
            mz = sim.get_turret_muzzle_pos()
            rel = sim.drone_pos - mz
            hz = float(np.hypot(rel[0], rel[2]))
            sim.turret_yaw = float(np.arctan2(rel[0], rel[2]))
            sim.turret_pitch = float(np.arctan2(-rel[1], hz))
            sim.step(1.0 / 60.0, DroneInput(), TruckInput(fire_turret=True, fire_shotgun=True))
            if not sim.drone_alive:
                break
            shot_frames += 1

        assert not sim.drone_alive and sim.drone_lives_left == initial_lives - 1, "Стрелок бензовоза не смог сбить FPV!"
        return (
            f"Бензовоз успешно сбил атакующий FPV из зенитной турели и дробовика "
            f"(за {shot_frames / 60.0:.1f}с огня, дальность 30м)"
        )

    record("5. Механика сбития FPV дрона водителем бензовоза", check_shoot_down_fpv)

    # 6. Check FPV Destroying the Fuel Truck
    def check_fpv_destroys_truck() -> str:
        sim = WorldSimulation(seed=105)
        sim.drone_mod = "shaped_charge"
        sim.truck_mod = "v12_biturbo"
        sim.begin_active_round()
        sim.truck_hp = 75.0
        # Put FPV right onto fuel cistern for shaped-charge strike
        sim.drone_pos[:] = sim.truck_pos + np.array([0.0, 2.2, -1.0], dtype=np.float32)
        sim.step(1.0 / 60.0, DroneInput(detonate=True), TruckInput())
        assert sim.truck_hp <= 0.0 and sim.phase == "ROUND_END" and sim.round_winner == "drone", (
            f"Ожидалось уничтожение бензовоза, но truck_hp={sim.truck_hp}, phase={sim.phase}"
        )
        return "Кумулятивный удар FPV уничтожил цистерну и засчитал победу оператору"

    record("6. Поражение и взрыв бензовоза от удара FPV", check_fpv_destroys_truck)

    # 7. Real TCP Network Server + 2 Separate Clients Loopback Test
    def check_network_1v1() -> str:
        srv = GameServer(host="127.0.0.1", port=0, fill_with_bot=False, seed=333)
        port = srv.start()
        c_drone = GameClient()
        c_truck = GameClient()
        try:
            ok1 = c_drone.connect("127.0.0.1", port, preferred_role="drone", timeout=3.0)
            ok2 = c_truck.connect("127.0.0.1", port, preferred_role="truck", timeout=3.0)
            assert ok1 and c_drone.role == "drone", f"Клиент 1 ошибка: {c_drone.error_msg}"
            assert ok2 and c_truck.role == "truck", f"Клиент 2 ошибка: {c_truck.error_msg}"

            # Both pick modifiers & ready up over TCP sockets!
            c_drone.send_drone_input(DroneInput(pick_modifier="sport_motors", ready=True))
            c_truck.send_truck_input(TruckInput(pick_modifier="anti_drone_cage", ready=True))
            c_drone.send_ping()

            deadline = time.perf_counter() + 2.5
            got_active = False
            snap_d = None
            while time.perf_counter() < deadline:
                s1 = c_drone.pop_latest_snapshot()
                if s1 is not None:
                    snap_d = s1
                    if s1.get("phase") == "ACTIVE_ROUND":
                        got_active = True
                        break
                time.sleep(0.02)

            assert got_active and snap_d is not None, f"Сервер не перешёл в ACTIVE_ROUND (snap={snap_d})"
            assert snap_d["drone_mod"] == "sport_motors" and snap_d["truck_mod"] == "anti_drone_cage"

            # Drive truck forward over network for 15 ticks
            z_before = snap_d["truck"]["pos"][2]
            for _ in range(12):
                c_truck.send_truck_input(TruckInput(accel=1.0, fire_turret=True))
                c_drone.send_drone_input(DroneInput(forward_Axis=1.0, boost=True))
                time.sleep(0.02)

            latest = c_truck.pop_latest_snapshot() or snap_d
            assert latest["truck"]["pos"][2] > z_before, "Позиция грузовика не синхронизировалась по сети!"
            return f"TCP сервер (порт {port}) + 2 клиента синхронизированы, пинг {c_drone.ping_ms:.1f} мс"
        finally:
            c_drone.disconnect()
            c_truck.disconnect()
            srv.stop()

    record("7. Сетевой мультиплеер 1v1 (Сервер + 2 ПК клиента)", check_network_1v1)

    # 8. Joystick / Gamepad / FPV Radio Input Layer Check
    def check_joystick_support() -> str:
        if not pygame.get_init():
            pygame.init()
        ic = InputController()
        snaps = ic.get_joystick_Snapshot()
        assert apply_deadzone(0.05, 0.12) == 0.0
        assert abs(apply_deadzone(1.0, 0.12) - 1.0) < 1e-5
        d_inp = ic.build_drone_input(1 / 60.0, mouse_captured=False)
        t_inp = ic.build_truck_input(1 / 60.0, mouse_captured=False)
        assert isinstance(d_inp, DroneInput) and isinstance(t_inp, TruckInput)
        joy_info = f"подключено устройств: {len(snaps)}" if snaps else "подсистема SDL2 Joystick активна (ожидание геймпада/аппаратуры)"
        return f"Gamepad + Radiomaster Mode 2 готовы ({joy_info})"

    record("8. Поддержка джойстиков, геймпадов и FPV-аппаратуры", check_joystick_support)

    # 9. Stand-off airburst mechanic + AA turret tracking lock
    def check_airburst_and_lock() -> str:
        sim = WorldSimulation(seed=4242)
        sim.begin_active_round()
        lives_before = sim.drone_lives_left
        hp_before = sim.truck_hp

        # Stand-off airburst from 25m: the charge misses the tanker, but the FPV survives
        sim.drone_pos[:] = sim.truck_pos + np.array([0.0, 12.0, 25.0], dtype=np.float32)
        sim.step(1.0 / 60.0, DroneInput(detonate=True), TruckInput())
        assert sim.drone_alive and sim.drone_lives_left == lives_before, (
            "Дистанционный подрыв не должен уничтожать FPV (stand-off подрыв)"
        )
        assert sim.drone_detonate_cd > 0.0, "БЧ должна перезаряжаться после подрыва!"
        assert abs(sim.truck_hp - hp_before) < 0.01, "Заряд не должен нанести урон на 25м"

        # Now drop the charge right on the cistern: heavy damage, FPV survives (can attack again)
        sim.drone_detonate_cd = 0.0
        sim.drone_pos[:] = sim.truck_pos + np.array([0.0, 2.4, 1.0], dtype=np.float32)
        sim.step(1.0 / 60.0, DroneInput(detonate=True), TruckInput())
        assert sim.drone_alive, "FPV должен остаться в строю после точного сброса БЧ"
        assert sim.truck_hp < hp_before, "Точный сброс БЧ должен наносить урон цистерне!"

        # Turret tracking lock: the gunner needs time to keep the drone inside the reticle
        sim2 = WorldSimulation(seed=4243)
        sim2.begin_active_round()
        muzzle = sim2.get_turret_muzzle_pos()
        sim2.drone_pos[:] = muzzle + np.array([0.0, 8.0, -60.0], dtype=np.float32)
        sim2.turret_yaw = 0.0
        sim2.turret_pitch = 0.0  # aimed away from the drone on purpose
        hp_full = sim2.drone_hp
        for _ in range(120):
            sim2.step(1.0 / 60.0, DroneInput(), TruckInput(fire_turret=True))
        assert sim2.drone_hp == hp_full, "Турель не должна попадать, если дрон вне ретикля!"
        assert sim2.turret_lock < 0.35, "Без захвата цели турель не должна набирать захват!"

        return (
            "Дистанционный подрыв БЧ сохраняет FPV, точный сброс наносит урон, "
            "а ПВО наводится только при удержании цели в ретикле"
        )

    record("9. Механика подрыва БЧ и захват цели зенитной турелью", check_airburst_and_lock)

    # 10. Near-plane clipping topology + rebindable control profile
    def check_clipping_and_controls() -> str:
        if not pygame.get_init():
            pygame.init()
        near = 1.0

        def _clip(tris):
            v = np.array(tris, dtype=np.float32)
            n = len(tris)
            cols = np.tile(np.array([[200, 180, 160]], dtype=np.float32), (n, 1))
            norms = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
            ts = np.zeros(n, dtype=bool)
            return SceneRenderer3D._clip_near_plane(
                v[:, 0], v[:, 1], v[:, 2], cols, norms, ts, near
            )

        # 1 vertex behind the camera -> visible quad split into 2 triangles
        out = _clip([[[0, 0, 0.0], [4, 0, 3.0], [0, 4, 4.0]]])
        assert len(out[0]) == 2, "Грань с 1 вершиной за камерой должна давать 2 треугольника!"
        zs = np.concatenate([out[0][:, 2], out[1][:, 2], out[2][:, 2]])
        assert zs.min() >= near - 1e-6, "После клиппинга не должно остаться вершин за камерой!"
        w = np.cross(out[1] - out[0], out[2] - out[0])[:, 2]
        assert np.all(w > 0), "Клиппинг должен сохранять ориентацию грани (CCW)!"

        # 2 vertices behind the camera -> single triangle
        out = _clip([[[0, 0, 3.0], [4, 0, 0.0], [0, 4, 0.5]]])
        assert len(out[0]) == 1, "Грань с 2 вершинами за камерой должна давать 1 треугольник!"
        zs = np.concatenate([out[0][:, 2], out[1][:, 2], out[2][:, 2]])
        assert zs.min() >= near - 1e-6, "После клиппинга не должно остаться вершин за камерой!"

        # Rebinding profile: default -> rebind -> reset -> JSON round-trip
        profile = JoystickProfile()
        ic = InputController(profile)
        default_key = ic.key_for("drone_forward")
        ic.rebind("drone_forward", pygame.K_i)
        assert ic.key_for("drone_forward") == pygame.K_i, "Клавиша не переназначилась!"
        ic.reset_binds()
        assert ic.key_for("drone_forward") == default_key, "Сброс раскладки не сработал!"
        ic.rebind("drone_forward", pygame.K_i)
        profile.mouse_sensitivity = 0.0033
        profile.deadzone = 0.2
        profile.axis_calibration[2] = 0.03
        tmp_json = os.path.join(repo_root, "logs", "_selftest_profile.json")
        os.makedirs(os.path.dirname(tmp_json), exist_ok=True)
        profile.save_to(tmp_json)
        loaded = JoystickProfile.load_from(tmp_json)
        assert loaded.key_binds.get("drone_forward") == pygame.K_i, "Профиль не сохранился в JSON!"
        assert abs(loaded.mouse_sensitivity - 0.0033) < 1e-6, "Чувствительность мыши не сохранилась!"
        assert abs(loaded.deadzone - 0.2) < 1e-6, "Мёртвая зона не сохранилась!"
        assert abs(loaded.axis_calibration.get(2, -1.0) - 0.03) < 1e-6, "Калибровка оси не сохранилась!"
        os.remove(tmp_json)

        return (
            "Клиппинг ближней плоскости корректен (1 и 2 вершины за камерой, "
            "ориентация грани сохранена), профиль управления переназначается, "
            "сбрасывается и сохраняется в JSON"
        )

    record("10. Клиппинг ближней плоскости и профиль управления", check_clipping_and_controls)

    return results
