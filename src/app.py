"""
Desktop GUI Application for Crimean Bridge 1v1 (FPV Drone vs Fuel Truck):
- Main Menu with live animated 3D Crimean Bridge background flyby
- Host 1v1 Server (for 2 PCs over LAN / Radmin VPN / ZeroTier / Port-Forward) with role choice
- Join 1v1 Server (type IP:Port of second PC or VPS + pick preferred role)
- Local Practice vs Smart Bot (as FPV Operator OR as Fuel Truck Driver/Gunner)
- Interactive Verification & Self-Test Mode ("Проверочный режим": automated diagnostics +
  live Dual-Viewport 3D Bot-vs-Bot Split-Screen showcase with instant weather/modifier hotkeys!)
- Live Joystick / Gamepad / Radiomaster FPV Radio Calibration & Tester Screen
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from .audio import SoundManager
from .bot_ai import BotDroneController, BotTruckController
from .config import DEFAULT_PORT, DRONE_MODIFIERS, DT, TRUCK_MODIFIERS, WEATHER_PRESETS
from .controls import (
    KEYBIND_LABELS_RU,
    REBINDABLE_ACTIONS,
    InputController,
    JoystickProfile,
    key_name,
)
from .network import GameClient, GameServer, get_local_lan_ip
from .physics import WorldSimulation
from .renderer import Camera3D, SceneRenderer3D
from .selftest import CheckResult, run_verification_suite


class DesktopGameApp:
    def __init__(self, repo_root: str, width: int = 1280, height: int = 720, start_mode: str = "menu") -> None:
        self.repo_root = repo_root
        pygame.init()
        pygame.joystick.init()
        pygame.display.set_caption("Крымский Мост 1v1 — Оператор FPV vs Бензовоз [Desktop 3D]")

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.running = True

        self.sim = WorldSimulation(seed=42)
        self.renderer = SceneRenderer3D(width, height, self.sim.barriers, repo_root)
        # Профиль управления (клавиши, чувствительность, мёртвая зона)
        # загружается из файла рядом с игрой и сохраняется при выходе.
        self.profile_path = os.path.join(repo_root, "logs", "input_profile.json")
        try:
            profile = JoystickProfile.load_from(self.profile_path)
        except Exception:
            profile = JoystickProfile()
        self.controls = InputController(profile)
        # Масштаб внутреннего разрешения мира из профиля (1.0 = натуральное)
        self.renderer.set_render_scale(profile.render_scale)
        self.audio = SoundManager(enabled=True)

        # App screens: "MENU", "HOST_SETUP", "JOIN_SETUP", "GAME_LOCAL", "GAME_NET", "VERIFY_MODE", "JOYSTICK_TEST"
        self.state = "MENU"
        self.menu_idx = 0
        self.menu_items = [
            ("Создать сервер 1 на 1 (Хост для игры с другом)", "HOST_SETUP"),
            ("Подключиться к серверу по IP (Второй ПК)", "JOIN_SETUP"),
            ("Тренировка за Оператора FPV (против ИИ Бензовоза)", "PRACTICE_DRONE"),
            ("Тренировка за Водителя Бензовоза (сбивать ИИ FPV)", "PRACTICE_TRUCK"),
            ("Проверочный режим (Автотесты + 3D Демо Сплит-Скрин)", "VERIFY_MODE"),
            ("Настройка Джойстика / Геймпада / FPV-аппаратуры", "JOYSTICK_TEST"),
            ("Назначение клавиш и калибровка (сохраняется в файл)", "KEYBIND_TEST"),
            ("Выход на рабочий стол", "EXIT"),
        ]

        # Networking fields
        self.ip_input: str = "127.0.0.1"
        self.port_input: str = str(DEFAULT_PORT)
        self.active_field: str = "ip"  # "ip" or "port"
        self.preferred_role: str = "drone"  # "drone" or "truck"
        self.host_bot_fill: bool = False
        self.status_banner: str = ""
        self.lan_ip: str = get_local_lan_ip()

        self.hosted_server: Optional[GameServer] = None
        self.net_client: Optional[GameClient] = None

        # Local practice & verify bots
        self.my_role: str = "drone"
        self.bot_drone = BotDroneController(difficulty=0.76)
        self.bot_truck = BotTruckController(difficulty=0.76)
        self.mouse_captured: bool = False

        # Verification mode state
        self.verify_results: List[CheckResult] = []
        self.verify_split_active: bool = False
        self._menu_flyby_t: float = 0.0
        # Экран назначения клавиш
        self.keybind_idx: int = 0
        self.keybind_list_idx: int = 0
        self.keybind_waiting: bool = False
        self.keybind_message: str = ""
        self.keybind_scroll: int = 0

        # Previous FX counts to trigger gunshot/explosion audio cues
        self._prev_explosions = 0
        self._prev_tracers = 0

        if start_mode == "verify":
            self._enter_verify_mode()

    # -----------------------------------------------------------------------
    # Mouse Capture Helper
    # -----------------------------------------------------------------------
    def _set_mouse_capture(self, capture: bool) -> None:
        self.mouse_captured = capture
        pygame.event.set_grab(capture)
        pygame.mouse.set_visible(not capture)
        if capture:
            pygame.mouse.get_rel()

    def _cleanup_network(self) -> None:
        self._set_mouse_capture(False)
        if self.net_client is not None:
            self.net_client.disconnect()
            self.net_client = None
        if self.hosted_server is not None:
            self.hosted_server.stop()
            self.hosted_server = None
        # Фоновые звуковые петли (ветер/ротор/мотор/дождь/РЭБ) должны
        # заглохнуть, как только игрок вышел из боя.
        try:
            self.audio.stop_all_loops()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                dt = min(self.clock.tick(60) / 1000.0, 0.05)
                self._process_events()
                self._update(dt)
                self._render()
                pygame.display.flip()
        finally:
            self._cleanup_network()
            # Сохраняем профиль управления (клавиши, чувствительность, зона)
            try:
                self._save_input_profile()
            except Exception:
                pass
            pygame.quit()

    # -----------------------------------------------------------------------
    # Event Dispatch
    # -----------------------------------------------------------------------
    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((max(800, event.w), max(520, event.h)), pygame.RESIZABLE)
                self.renderer.resize(self.screen.get_width(), self.screen.get_height())

            self.controls.handle_event(event)

            if self.state == "MENU":
                self._handle_menu_event(event)
            elif self.state in ("HOST_SETUP", "JOIN_SETUP"):
                self._handle_net_setup_event(event)
            elif self.state in ("GAME_LOCAL", "GAME_NET"):
                self._handle_gameplay_event(event)
            elif self.state == "VERIFY_MODE":
                self._handle_verify_event(event)
            elif self.state == "JOYSTICK_TEST":
                self._handle_joystick_event(event)
            elif self.state == "KEYBIND_TEST":
                self._handle_keybind_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.menu_idx = (self.menu_idx - 1) % len(self.menu_items)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.menu_idx = (self.menu_idx + 1) % len(self.menu_items)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._activate_menu_choice(self.menu_items[self.menu_idx][1])
            elif event.key == pygame.K_ESCAPE:
                self.running = False

        elif event.type == pygame.JOYHATMOTION:
            _, hy = event.value
            if hy > 0:
                self.menu_idx = (self.menu_idx - 1) % len(self.menu_items)
            elif hy < 0:
                self.menu_idx = (self.menu_idx + 1) % len(self.menu_items)

        elif event.type == pygame.JOYBUTTONDOWN and event.button in (0, 7):
            self._activate_menu_choice(self.menu_items[self.menu_idx][1])

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            w, _ = self.screen.get_size()
            card_w = min(680, w - 80)
            card_x = w // 2 - card_w // 2
            for i, (_, action) in enumerate(self.menu_items):
                cy = 210 + i * 62
                if card_x <= mx <= card_x + card_w and cy <= my <= cy + 52:
                    self.menu_idx = i
                    self._activate_menu_choice(action)
                    break

    def _activate_menu_choice(self, action: str) -> None:
        self.status_banner = ""
        if action == "EXIT":
            self.running = False
        elif action == "HOST_SETUP":
            self.lan_ip = get_local_lan_ip()
            self.active_field = "port"
            self.state = "HOST_SETUP"
        elif action == "JOIN_SETUP":
            self.active_field = "ip"
            self.state = "JOIN_SETUP"
        elif action == "PRACTICE_DRONE":
            self._start_local_practice("drone")
        elif action == "PRACTICE_TRUCK":
            self._start_local_practice("truck")
        elif action == "VERIFY_MODE":
            self._enter_verify_mode()
        elif action == "JOYSTICK_TEST":
            self.controls.refresh_joysticks()
            self.state = "JOYSTICK_TEST"
        elif action == "KEYBIND_TEST":
            self.keybind_idx = 0
            self.keybind_list_idx = 0
            self.keybind_waiting = False
            self.keybind_message = ""
            self.state = "KEYBIND_TEST"

    def _start_local_practice(self, role: str) -> None:
        self._cleanup_network()
        self.my_role = role
        self.sim = WorldSimulation()
        self.state = "GAME_LOCAL"
        self._set_mouse_capture(False)

    def _handle_net_setup_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.state = "MENU"
            return
        if event.key == pygame.K_TAB:
            if self.state == "JOIN_SETUP":
                self.active_field = "port" if self.active_field == "ip" else "ip"
            return
        if event.key == pygame.K_F1:
            self.preferred_role = "drone"
            return
        if event.key == pygame.K_F2:
            self.preferred_role = "truck"
            return
        if event.key == pygame.K_F3 and self.state == "HOST_SETUP":
            self.host_bot_fill = not self.host_bot_fill
            return

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            try:
                port = int(self.port_input.strip() or str(DEFAULT_PORT))
            except ValueError:
                self.status_banner = "Некорректный порт! Введите число (например 27015)."
                return

            if self.state == "HOST_SETUP":
                self._launch_hosted_match(port)
            else:
                self._join_remote_match(self.ip_input.strip() or "127.0.0.1", port)
            return

        # Text editing for IP / Port
        target = "port" if self.state == "HOST_SETUP" else self.active_field
        if event.key == pygame.K_BACKSPACE:
            if target == "ip":
                self.ip_input = self.ip_input[:-1]
            else:
                self.port_input = self.port_input[:-1]
        else:
            ch = event.unicode
            if ch and ch in "0123456789.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-":
                if target == "ip" and len(self.ip_input) < 42:
                    self.ip_input += ch
                elif target == "port" and ch.isdigit() and len(self.port_input) < 5:
                    self.port_input += ch

    def _launch_hosted_match(self, port: int) -> None:
        self._cleanup_network()
        try:
            srv = GameServer(host="0.0.0.0", port=port, fill_with_bot=self.host_bot_fill)
            bound = srv.start()
            cli = GameClient()
            if not cli.connect("127.0.0.1", bound, preferred_role=self.preferred_role, timeout=4.0):
                srv.stop()
                self.status_banner = cli.error_msg
                return
            self.hosted_server = srv
            self.net_client = cli
            self.my_role = cli.role
            self.sim = WorldSimulation()
            self.state = "GAME_NET"
        except OSError as exc:
            self.status_banner = f"Ошибка запуска сервера на порту {port}: {exc}"

    def _join_remote_match(self, host: str, port: int) -> None:
        self._cleanup_network()
        cli = GameClient()
        if not cli.connect(host, port, preferred_role=self.preferred_role, timeout=5.0):
            self.status_banner = cli.error_msg
            return
        self.net_client = cli
        self.my_role = cli.role
        self.sim = WorldSimulation()
        self.state = "GAME_NET"

    def _handle_gameplay_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.mouse_captured:
                    self._set_mouse_capture(False)
                else:
                    self._cleanup_network()
                    self.state = "MENU"
            elif event.key == pygame.K_TAB:
                self._set_mouse_capture(not self.mouse_captured)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.sim.phase == "ACTIVE_ROUND" and not self.mouse_captured:
                self._set_mouse_capture(True)

    def _enter_verify_mode(self) -> None:
        self._cleanup_network()
        self.state = "VERIFY_MODE"
        self.verify_results = run_verification_suite(self.repo_root)
        # Also spin up a live Split-Screen 3D Bot-vs-Bot combat simulation so the user can visually inspect!
        self.sim = WorldSimulation(seed=2026)
        self.sim.drone_mod = "shaped_charge"
        self.sim.truck_mod = "twin_zu_turret"
        self.sim.begin_active_round()
        self.verify_split_active = True

    def _handle_verify_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = "MENU"
            elif event.key == pygame.K_r:
                self._enter_verify_mode()
            elif event.key == pygame.K_w:
                # Cycle weather live during verification split-screen!
                cur_idx = next((i for i, w in enumerate(WEATHER_PRESETS) if w.id == self.sim.weather_id), 0)
                self.sim.weather_id = WEATHER_PRESETS[(cur_idx + 1) % len(WEATHER_PRESETS)].id
                self.sim.last_event_ru = f"[Проверка] Принудительная смена погоды: {self.sim.weather.name_ru}"
            elif event.key == pygame.K_SPACE:
                self.verify_split_active = not self.verify_split_active

    def _handle_keybind_event(self, event: pygame.event.Event) -> None:
        """Экран назначения клавиш: выбор действия -> нажатие новой клавиши."""
        if event.type != pygame.KEYDOWN:
            return
        if self.keybind_waiting:
            # Ждём нажатия клавиши для назначения
            if event.key == pygame.K_ESCAPE:
                self.keybind_waiting = False
                self.keybind_message = "Назначение отменено"
                return
            action = REBINDABLE_ACTIONS[self.keybind_idx]
            self.controls.rebind(action, event.key)
            self.keybind_waiting = False
            self.keybind_message = (
                f"{KEYBIND_LABELS_RU.get(action, action)} -> {key_name(event.key)}"
            )
            return

        if event.key == pygame.K_ESCAPE:
            self.state = "MENU"
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.keybind_idx = (self.keybind_idx + 1) % len(REBINDABLE_ACTIONS)
        elif event.key in (pygame.K_UP, pygame.K_w):
            self.keybind_idx = (self.keybind_idx - 1) % len(REBINDABLE_ACTIONS)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.keybind_waiting = True
            self.keybind_message = "Нажмите новую клавишу (ESC - отмена)"
        elif event.key == pygame.K_BACKSPACE:
            self.controls.reset_binds()
            self.keybind_message = "Восстановлена стандартная раскладка"
        elif event.key == pygame.K_F2:
            self._save_input_profile()
            self.keybind_message = "Профиль сохранён"

    def _save_input_profile(self) -> bool:
        """Сохраняет профиль управления (клавиши, чувствительность) в JSON."""
        try:
            ok = self.controls.profile.save_to(self.profile_path)
        except Exception:
            ok = False
        return ok

    def _handle_joystick_event(self, event: pygame.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = "MENU"
            elif event.key == pygame.K_r:
                self.controls.refresh_joysticks()
            elif event.key == pygame.K_LEFTBRACKET:
                self.controls.profile.deadzone = max(0.02, round(self.controls.profile.deadzone - 0.02, 2))
            elif event.key == pygame.K_RIGHTBRACKET:
                self.controls.profile.deadzone = min(0.40, round(self.controls.profile.deadzone + 0.02, 2))
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._change_render_scale(-0.1)
            elif event.key in (pygame.K_EQUALS, pygame.K_KP_EQUALS):
                self._change_render_scale(+0.1)
            elif event.key == pygame.K_c:
                # Калибровка центра стиков: отпустите все стики и нажмите C
                n = self.controls.calibrate_active_axes()
                if n > 0:
                    self.status_banner = f"Центр стиков откалиброван по {n} осям (стики должны быть отпущены!)"
                else:
                    self.status_banner = "Нет подключённых устройств - калибровать нечего"

    def _change_render_scale(self, delta: float) -> None:
        """Меняет целевой масштаб внутреннего разрешения мира (0.5..1.0)."""
        prof = self.controls.profile
        prof.render_scale = round(max(0.5, min(1.0, prof.render_scale + delta)), 2)
        self.renderer.set_render_scale(prof.render_scale)
        self.status_banner = (
            f"Масштаб рендера: {int(prof.render_scale * 100)}% "
            f"(FPS-стабилизация сама подстроит точное значение)"
        )

    # -----------------------------------------------------------------------
    # Frame Update
    # -----------------------------------------------------------------------
    def _update(self, dt: float) -> None:
        if self.state == "MENU":
            self._menu_flyby_t += dt
            return

        if self.state == "GAME_LOCAL":
            if self.sim.phase != "ACTIVE_ROUND" and self.mouse_captured:
                self._set_mouse_capture(False)
            elif self.sim.phase == "ACTIVE_ROUND" and not self.mouse_captured:
                # Auto-grab cursor when round goes live
                self._set_mouse_capture(True)

            if self.my_role == "drone":
                d_in = self.controls.build_drone_input(dt, self.mouse_captured)
                t_in = self.bot_truck.compute_input(self.sim, dt)
            else:
                d_in = self.bot_drone.compute_input(self.sim, dt)
                t_in = self.controls.build_truck_input(dt, self.mouse_captured)

            self.sim.step(dt, d_in, t_in)
            self._trigger_audio_cues()
            self._update_ambient_audio()

        elif self.state == "GAME_NET":
            if self.net_client is None or not self.net_client.connected:
                err = self.net_client.error_msg if self.net_client else "Отключено"
                self._cleanup_network()
                self.status_banner = err
                self.state = "MENU"
                return

            snap = self.net_client.pop_latest_snapshot()
            if snap is not None:
                self.sim.apply_snapshot(snap)

            if self.sim.phase != "ACTIVE_ROUND" and self.mouse_captured:
                self._set_mouse_capture(False)

            if self.my_role == "drone":
                d_in = self.controls.build_drone_input(dt, self.mouse_captured)
                self.net_client.send_drone_input(d_in)
            else:
                t_in = self.controls.build_truck_input(dt, self.mouse_captured)
                self.net_client.send_truck_input(t_in)

            self._trigger_audio_cues()
            self._update_ambient_audio()

        elif self.state == "VERIFY_MODE" and self.verify_split_active:
            d_in = self.bot_drone.compute_input(self.sim, dt)
            t_in = self.bot_truck.compute_input(self.sim, dt)
            self.sim.step(dt, d_in, t_in)

    def _trigger_audio_cues(self) -> None:
        if len(self.sim.explosions) > self._prev_explosions:
            self.audio.play("explosion")
        elif len(self.sim.tracers) > self._prev_tracers:
            if len(self.sim.tracers) - self._prev_tracers >= 5:
                self.audio.play("shotgun")
            else:
                self.audio.play("turret")
        self._prev_explosions = len(self.sim.explosions)
        self._prev_tracers = len(self.sim.tracers)

    def _update_ambient_audio(self) -> None:
        """
        Управляет фоновыми звуковыми циклами (мотоцикл/ветер/дождь/РЭБ).

        Громкость зависит от погоды, состояния дрона и расстояния до
        бензовоза: слышно ветер, шорох дождя и треск помех от РЭБ-купола.
        """
        w = self.sim.weather
        active = self.sim.phase == "ACTIVE_ROUND"

        # --- Ветер: сильнее в шторм и на высоких горах ----------------------------
        wind_v = 0.10 + 0.055 * float(w.gust_strength)
        if active and self.my_role == "drone":
            speed = float(np.linalg.norm(self.sim.drone_vel))
            wind_v += min(0.28, speed * 0.012)
        if active:
            self.audio.start_loop("wind", wind_v)
        else:
            self.audio.stop_loop("wind")

        # --- Роторы дрона ------------------------------------------------------
        rotor_v = 0.0
        if active and self.my_role == "drone" and self.sim.drone_alive:
            rotor_v = 0.20 + (0.10 if self.sim.drone_vel.size else 0.0)
            rotor_v = 0.26
        elif active and self.my_role == "truck":
            # У водителя слышен звук чужого дрона - тем громче, чем ближе
            dist = float(np.linalg.norm(self.sim.drone_pos - self.sim.truck_pos))
            rotor_v = max(0.0, 0.30 - dist / 260.0)
        if rotor_v > 0.01:
            self.audio.start_loop("rotor", rotor_v)
        else:
            self.audio.stop_loop("rotor")

        # --- Мотор бензовоза ----------------------------------------------------
        engine_v = 0.0
        if active and self.my_role == "truck":
            engine_v = 0.24
        elif active and self.my_role == "drone":
            dist = float(np.linalg.norm(self.sim.drone_pos - self.sim.truck_pos))
            engine_v = max(0.0, 0.26 - dist / 300.0)
        if engine_v > 0.01:
            self.audio.start_loop("engine", engine_v)
        else:
            self.audio.stop_loop("engine")

        # --- Дождь / снег -------------------------------------------------------
        rain_v = 0.0
        if active and float(w.rain_intensity) > 0.05:
            rain_v = 0.06 + 0.16 * float(w.rain_intensity)
        if rain_v > 0.01:
            self.audio.start_loop("rain", rain_v)
        else:
            self.audio.stop_loop("rain")

        # --- Помехи от РЭБ-купола ----------------------------------------------
        ew_v = 0.0
        if active and self.sim.ew_active_timer > 0.0:
            if self.my_role == "drone":
                dist = float(np.linalg.norm(self.sim.drone_pos - self.sim.truck_pos))
                ew_v = max(0.0, 0.22 - dist / 320.0)
            else:
                ew_v = 0.05
        if ew_v > 0.01:
            self.audio.start_loop("ew_static", ew_v)
        else:
            self.audio.stop_loop("ew_static")

    # -----------------------------------------------------------------------
    # Rendering Screens
    # -----------------------------------------------------------------------
    def _render(self) -> None:
        if self.state == "MENU":
            self._render_main_menu()
        elif self.state in ("HOST_SETUP", "JOIN_SETUP"):
            self._render_net_setup()
        elif self.state in ("GAME_LOCAL", "GAME_NET"):
            self._render_gameplay()
        elif self.state == "VERIFY_MODE":
            self._render_verify_mode()
        elif self.state == "JOYSTICK_TEST":
            self._render_joystick_screen()
        elif self.state == "KEYBIND_TEST":
            self._render_keybind_screen()

    def _render_background_flyby(self) -> None:
        # Orbit camera smoothly around the iconic white Crimean Bridge arches
        ang = self._menu_flyby_t * 0.22
        radius = 165.0
        cam = Camera3D(
            pos=np.array([math.sin(ang) * radius, 48.0 + math.sin(ang * 1.7) * 8.0, math.cos(ang) * radius], dtype=np.float32),
            yaw=ang + math.pi,
            pitch=0.14,
            roll=0.0,
            fov_deg=78.0,
        )
        self.renderer.render_world(self.screen, self.sim, cam, hide_own_drone=False)
        dim = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        dim.fill((8, 14, 24, 155))
        self.screen.blit(dim, (0, 0))

    def _render_main_menu(self) -> None:
        self._render_background_flyby()
        w, h = self.screen.get_size()
        cx = w // 2

        title = self.renderer.font_xl.render("КРЫМСКИЙ МОСТ 1v1", True, (255, 220, 95))
        sub = self.renderer.font_md.render(
            "3D Дуэль по сети: Оператор FPV Дрона  против  Водителя-Стрелка Бензовоза",
            True,
            (185, 225, 255),
        )
        self.screen.blit(title, (cx - title.get_width() // 2, 58))
        self.screen.blit(sub, (cx - sub.get_width() // 2, 118))

        card_w = min(680, w - 80)
        card_x = cx - card_w // 2
        for i, (label, _) in enumerate(self.menu_items):
            cy = 198 + i * 62
            sel = i == self.menu_idx
            card = pygame.Surface((card_w, 52), pygame.SRCALPHA)
            card.fill((38, 86, 138, 235) if sel else (18, 26, 38, 210))
            self.screen.blit(card, (card_x, cy))
            pygame.draw.rect(
                self.screen,
                (255, 212, 75) if sel else (68, 88, 112),
                (card_x, cy, card_w, 52),
                2,
            )
            prefix = "> " if sel else "  "
            txt = self.renderer.font_md.render(prefix + label, True, (255, 255, 255) if sel else (205, 218, 232))
            self.screen.blit(txt, (card_x + 18, cy + 13))

        if self.status_banner:
            err = self.renderer.font_sm.render(self.status_banner, True, (255, 100, 90))
            self.screen.blit(err, (cx - err.get_width() // 2, h - 64))

        joy_cnt = len(self.controls.get_joystick_Snapshot())
        footer = self.renderer.font_xs.render(
            f"Управление: Мышь+Клавиатура / Геймпад / FPV-Пульт (обнаружено джойстиков: {joy_cnt})  |  Локальный IP: {self.lan_ip}",
            True,
            (165, 185, 208),
        )
        self.screen.blit(footer, (cx - footer.get_width() // 2, h - 32))

    def _render_net_setup(self) -> None:
        self._render_background_flyby()
        w, h = self.screen.get_size()
        cx = w // 2
        is_host = self.state == "HOST_SETUP"

        panel = pygame.Surface((760, 470), pygame.SRCALPHA)
        panel.fill((12, 18, 30, 232))
        px, py = cx - 380, h // 2 - 235
        self.screen.blit(panel, (px, py))
        pygame.draw.rect(self.screen, (90, 155, 220), (px, py, 760, 470), 2)

        hdr_txt = "СОЗДАНИЕ СЕРВЕРА 1 НА 1 (ХОСТ)" if is_host else "ПОДКЛЮЧЕНИЕ К СЕРВЕРУ (2-Й КОМПЬЮТЕР)"
        hdr = self.renderer.font_lg.render(hdr_txt, True, (255, 218, 90))
        self.screen.blit(hdr, (cx - hdr.get_width() // 2, py + 24))

        y = py + 86
        if is_host:
            info = self.renderer.font_sm.render(
                f"Сообщите другу ваш IP-адрес (LAN / Radmin VPN / ZeroTier):  {self.lan_ip}",
                True,
                (125, 235, 165),
            )
            self.screen.blit(info, (px + 36, y))
            y += 46
        else:
            ip_sel = self.active_field == "ip"
            lbl = self.renderer.font_md.render(
                f"IP-адрес Хоста:  [{self.ip_input}{'_' if ip_sel else ''}]",
                True,
                (255, 230, 110) if ip_sel else (215, 225, 238),
            )
            self.screen.blit(lbl, (px + 36, y))
            y += 46

        port_sel = is_host or self.active_field == "port"
        plbl = self.renderer.font_md.render(
            f"TCP Порт игры:   [{self.port_input}{'_' if port_sel else ''}]   (TAB — переключить поле)",
            True,
            (255, 230, 110) if port_sel else (215, 225, 238),
        )
        self.screen.blit(plbl, (px + 36, y))
        y += 54

        role_str = "ОПЕРАТОР FPV ДРОНА [F1]" if self.preferred_role == "drone" else "ВОДИТЕЛЬ-СТРЕЛОК БЕНЗОВОЗА [F2]"
        rlbl = self.renderer.font_md.render(f"Ваша роль (F1 / F2):  {role_str}", True, (115, 215, 255))
        self.screen.blit(rlbl, (px + 36, y))
        y += 44

        if is_host:
            bf_txt = "ВКЛ (можно играть сразу пока друг заходит)" if self.host_bot_fill else "ВЫКЛ (ждём подключения 2-го ПК в лобби)"
            blbl = self.renderer.font_sm.render(f"[F3] Заполнение ИИ-ботом до входа 2-го игрока: {bf_txt}", True, (205, 215, 230))
            self.screen.blit(blbl, (px + 36, y))
            y += 44

        go_txt = "Нажмите [ENTER], чтобы запустить сервер и войти в матч!" if is_host else "Нажмите [ENTER], чтобы подключиться к серверу!"
        go_s = self.renderer.font_md.render(go_txt, True, (95, 245, 135))
        self.screen.blit(go_s, (cx - go_s.get_width() // 2, py + 365))

        esc_s = self.renderer.font_xs.render("[ESC] — Назад в меню  |  Инструкция по хостингу: см. файл HOSTING_RU.md", True, (165, 180, 205))
        self.screen.blit(esc_s, (cx - esc_s.get_width() // 2, py + 422))

        if self.status_banner:
            err = self.renderer.font_sm.render(self.status_banner, True, (255, 95, 85))
            self.screen.blit(err, (cx - err.get_width() // 2, py + 322))

    def _render_gameplay(self) -> None:
        if self.my_role == "drone":
            cam = self.renderer.make_drone_camera(self.sim)
            proj = self.renderer.render_world(self.screen, self.sim, cam, hide_own_drone=True)
            self.renderer.draw_drone_hud(self.screen, self.sim, proj)
        else:
            cam = self.renderer.make_truck_camera(self.sim)
            proj = self.renderer.render_world(self.screen, self.sim, cam, hide_own_drone=False)
            self.renderer.draw_truck_hud(self.screen, self.sim, proj)

        net_info = ""
        if self.state == "GAME_NET" and self.net_client is not None:
            roles_str = ", ".join(self.net_client.online_roles) if self.net_client.online_roles else self.my_role
            host_hint = f" | Ваш IP для друга: {self.lan_ip}:{self.hosted_server.bound_port}" if self.hosted_server else ""
            net_info = f"Сеть: онлайн [{roles_str}] | Пинг: {self.net_client.ping_ms:.0f} мс{host_hint}"
            ping_lbl = self.renderer.font_xs.render(net_info, True, (150, 220, 255))
            self.screen.blit(ping_lbl, (self.screen.get_width() - ping_lbl.get_width() - 16, 48))

        if self.sim.phase != "ACTIVE_ROUND":
            self.renderer.draw_phase_overlay(self.screen, self.sim, self.my_role, net_status=net_info)
        elif not self.mouse_captured:
            hint = self.renderer.font_sm.render(
                "Кликните ЛКМ или нажмите [TAB] для захвата мыши  |  [ESC] — Выход в меню",
                True,
                (255, 230, 120),
            )
            self.screen.blit(hint, (self.screen.get_width() // 2 - hint.get_width() // 2, 76))

    def _render_verify_mode(self) -> None:
        w, h = self.screen.get_size()
        self.screen.fill((12, 16, 24))

        # Top Half: Live Split-Screen 3D Combat (Left = FPV Drone View, Right = Fuel Truck AA Gunner View)
        split_h = max(250, int(h * 0.54))
        half_w = w // 2 - 4

        left_surf = pygame.Surface((half_w, split_h))
        right_surf = pygame.Surface((half_w, split_h))

        cam_d = self.renderer.make_drone_camera(self.sim)
        pt = self.renderer.render_world(left_surf, self.sim, cam_d, hide_own_drone=True)
        self.renderer.draw_drone_hud(left_surf, self.sim, pt)

        cam_t = self.renderer.make_truck_camera(self.sim)
        pd = self.renderer.render_world(right_surf, self.sim, cam_t, hide_own_drone=False)
        self.renderer.draw_truck_hud(right_surf, self.sim, pd)

        self.screen.blit(left_surf, (2, 2))
        self.screen.blit(right_surf, (half_w + 6, 2))
        pygame.draw.line(self.screen, (255, 215, 80), (half_w + 3, 0), (half_w + 3, split_h), 3)

        # Bottom Half: Automated Verification Checklist Report
        y0 = split_h + 10
        passed_cnt = sum(1 for r in self.verify_results if r.passed)
        total_cnt = len(self.verify_results)
        hdr_col = (85, 245, 130) if passed_cnt == total_cnt else (255, 95, 80)

        hdr = self.renderer.font_md.render(
            f"ПРОВЕРОЧНЫЙ РЕЖИМ РАБОТОСПОСОБНОСТИ: ПРОЙДЕНО {passed_cnt}/{total_cnt} АВТОТЕСТОВ",
            True,
            hdr_col,
        )
        self.screen.blit(hdr, (18, y0))

        ctrl = self.renderer.font_xs.render(
            "[W] Сменить погоду  |  [SPACE] Пауза боя  |  [R] Перезапустить тесты  |  [ESC] В меню",
            True,
            (185, 208, 235),
        )
        self.screen.blit(ctrl, (18, y0 + 26))

        row_y = y0 + 50
        row_h = max(18, (h - row_y - 8) // max(total_cnt, 1))
        max_line_w = w - 40
        for r in self.verify_results:
            mark = "[OK]" if r.passed else "[FAIL]"
            mcol = (90, 240, 135) if r.passed else (255, 85, 80)
            line = self.renderer.font_xs.render(
                f"{mark} {r.name_ru} — {r.details_ru} ({r.duration_ms:.0f} мс)",
                True,
                mcol,
            )
            self.screen.blit(line, (22, row_y), (0, 0, min(line.get_width(), max_line_w), line.get_height()))
            row_y += row_h

    def _render_keybind_screen(self) -> None:
        """Экран назначения клавиш и калибровки управления."""
        w, h = self.screen.get_size()
        self.screen.fill((14, 20, 32))
        cx = w // 2

        title = self.renderer.font_lg.render("НАЗНАЧЕНИЕ КЛАВИШ И КАЛИБРОВКА", True, (255, 218, 90))
        self.screen.blit(title, (cx - title.get_width() // 2, 26))

        prof = self.controls.profile
        sub = self.renderer.font_sm.render(
            f"Мышь: {prof.mouse_sensitivity:.4f}   |   Геймпад/пульт: {prof.look_sensitivity:.2f}   |   "
            f"Мёртвая зона: {prof.deadzone:.2f}   |   Режим FPV Mode 2: "
            f"{'вкл' if prof.fpv_radio_mode2 else 'выкл'}",
            True,
            (130, 215, 255),
        )
        self.screen.blit(sub, (cx - sub.get_width() // 2, 68))

        # --- Список действий ---
        rows_h = 26
        box_h = min(h - 220, rows_h * len(REBINDABLE_ACTIONS) + 16)
        box_y = 100
        box = pygame.Surface((min(760, w - 80), box_h), pygame.SRCALPHA)
        box.fill((22, 32, 48, 232))
        self.screen.blit(box, (cx - box.get_width() // 2, box_y))
        pygame.draw.rect(self.screen, (86, 140, 200), (cx - box.get_width() // 2, box_y, box.get_width(), box_h), 2)

        visible = max(1, (box_h - 12) // rows_h)
        if self.keybind_idx < self.keybind_scroll:
            self.keybind_scroll = self.keybind_idx
        elif self.keybind_idx >= self.keybind_scroll + visible:
            self.keybind_scroll = self.keybind_idx - visible + 1
        self.keybind_scroll = max(0, min(self.keybind_scroll, len(REBINDABLE_ACTIONS) - visible))

        for i in range(visible):
            ai = self.keybind_scroll + i
            if ai >= len(REBINDABLE_ACTIONS):
                break
            action = REBINDABLE_ACTIONS[ai]
            sel = ai == self.keybind_idx
            ry = box_y + 8 + i * rows_h
            if sel:
                hl = pygame.Surface((box.get_width() - 12, rows_h - 2), pygame.SRCALPHA)
                hl.fill((52, 104, 168, 235))
                self.screen.blit(hl, (cx - box.get_width() // 2 + 6, ry))
            label = KEYBIND_LABELS_RU.get(action, action)
            lbl = self.renderer.font_xs.render(label, True, (255, 255, 255) if sel else (198, 212, 230))
            self.screen.blit(lbl, (cx - box.get_width() // 2 + 16, ry + 4))
            key_txt = key_name(self.controls.key_for(action))
            kcol = (255, 205, 90) if sel else (150, 200, 235)
            kw = self.renderer.font_xs.render(key_txt, True, kcol)
            self.screen.blit(kw, (cx + box.get_width() // 2 - kw.get_width() - 20, ry + 4))

        # --- Подсказки ---
        info = self.keybind_message or (
            "[W/S] или [UP/DOWN] - выбрать действие   |   [ENTER] - назначить новую клавишу   |   "
            "[BACKSPACE] - сбросить всё   |   [F2] - сохранить профиль   |   [ESC] - назад"
        )
        info_s = self.renderer.font_xs.render(info, True, (255, 225, 130) if self.keybind_waiting else (195, 210, 228))
        self.screen.blit(info_s, (cx - info_s.get_width() // 2, h - 96))

        save_txt = "Профиль: " + self.profile_path
        st = self.renderer.font_xs.render(save_txt, True, (150, 170, 195))
        self.screen.blit(st, (cx - st.get_width() // 2, h - 68))

        hint = self.renderer.font_xs.render(
            "Мышь всегда отвечает за курс/тангаж; геймпад и FPV-пульт (Mode 2) работают параллельно с клавишами.",
            True,
            (170, 190, 215),
        )
        self.screen.blit(hint, (cx - hint.get_width() // 2, h - 44))

    def _render_joystick_screen(self) -> None:
        w, h = self.screen.get_size()
        self.screen.fill((14, 20, 32))
        cx = w // 2

        title = self.renderer.font_lg.render("НАСТРОЙКА И ТЕСТ ДЖОЙСТИКА / ГЕЙМПАДА / FPV-ПУЛЬТА", True, (255, 218, 90))
        self.screen.blit(title, (cx - title.get_width() // 2, 28))

        mode_str = "FPV Аппаратура Mode 2 (Radiomaster / EdgeTX USB)" if self.controls.profile.fpv_radio_mode2 else "Стандартный Геймпад (Xbox / DualSense / DirectInput)"
        sub = self.renderer.font_sm.render(
            f"Профиль осей [клавиша M]: {mode_str}   |   Мёртвая зона [ / ]: {self.controls.profile.deadzone:.2f}",
            True,
            (130, 215, 255),
        )
        self.screen.blit(sub, (cx - sub.get_width() // 2, 74))

        scale_pct = int(round(self.controls.profile.render_scale * 100))
        scale_txt = self.renderer.font_sm.render(
            f"Качество 3D-картинки [- / =]: {scale_pct}%  "
            f"(меньше - быстрее на слабых ПК, прицел и HUD всегда чёткие)",
            True,
            (255, 205, 120),
        )
        self.screen.blit(scale_txt, (cx - scale_txt.get_width() // 2, 100))

        snaps = self.controls.get_joystick_Snapshot()
        if not snaps:
            empty_lines = [
                "Подключенные контроллеры сейчас не обнаружены.",
                "Подключите USB/Bluetooth геймпад или пульт — подхватится на лету!",
            ]
            for li, line in enumerate(empty_lines):
                et = self.renderer.font_md.render(line, True, (255, 185, 95))
                self.screen.blit(et, (cx - et.get_width() // 2, 160 + li * 30))
        else:
            y = 126
            for js in snaps:
                box = pygame.Surface((w - 80, 175), pygame.SRCALPHA)
                box.fill((24, 34, 52, 230))
                self.screen.blit(box, (40, y))
                pygame.draw.rect(self.screen, (95, 175, 240), (40, y, w - 80, 175), 2)

                nm = self.renderer.font_md.render(
                    f"Устройство #{js.index}: {js.name}  (Осей: {js.num_axes}, Кнопок: {js.num_buttons}, Hat: {js.num_hats})",
                    True,
                    (110, 245, 155),
                )
                self.screen.blit(nm, (58, y + 12))

                # Draw live axis bars
                for ai, aval in enumerate(js.axes[:8]):
                    bx = 58 + (ai % 4) * 285
                    by = y + 52 + (ai // 4) * 42
                    center = self.controls.profile.axis_calibration.get(ai)
                    if center is None:
                        lbl = self.renderer.font_xs.render(f"Ось {ai}: {aval:+.2f}", True, (220, 230, 242))
                    else:
                        lbl = self.renderer.font_xs.render(
                            f"Ось {ai}: {aval:+.2f} (центр {center:+.2f})", True, (220, 230, 242)
                        )
                    self.screen.blit(lbl, (bx, by))
                    pygame.draw.rect(self.screen, (50, 62, 82), (bx + 92, by + 2, 160, 14))
                    fill_x = bx + 92 + 80
                    bar_w = int(aval * 80)
                    if bar_w >= 0:
                        pygame.draw.rect(self.screen, (90, 215, 255), (fill_x, by + 2, bar_w, 14))
                    else:
                        pygame.draw.rect(self.screen, (255, 155, 75), (fill_x + bar_w, by + 2, -bar_w, 14))

                pressed = [str(bi) for bi, bval in enumerate(js.buttons) if bval]
                btn_txt = f"Нажатые кнопки: {', '.join(pressed) if pressed else 'нет'}   |   Крестовина (Hat): {js.hats}"
                self.screen.blit(self.renderer.font_xs.render(btn_txt, True, (255, 225, 120)), (58, y + 140))
                y += 190

        # Reference mapping guide at bottom
        guide = [
            "Раскладка Геймпада для FPV Дрона: Левый стик — крен/ход, Правый стик — курс/тангаж, RT/LT — набор/снижение, LB — Форсаж, RB/A — Подрыв БЧ",
            "Раскладка Геймпада для Бензовоза: Левый стик — руль, RT/LT — газ/тормоз, Правый стик — прицел ПВО, LB — ЗПУ Турель, RB/B — Дробовик, X — РЭБ, Y — Дым",
            "[M] Переключить режим FPV-Аппаратуры (Mode 2)   |   [R] Обновить USB устройства   |   [- / =] Качество 3D-картинки   |   [ESC] Вернуться в главное меню",
            "[C] Откалибровать центр стиков (отпустите стики и нажмите C) - значения осей в покое запоминаются в профиль",
        ]
        for gi, gline in enumerate(guide):
            gs = self.renderer.font_xs.render(gline, True, (195, 210, 228))
            self.screen.blit(gs, (cx - gs.get_width() // 2, h - 92 + gi * 26))
