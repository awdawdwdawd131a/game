"""
Controller, Gamepad, Flight-Stick & RC FPV Transmitter (ExpressLRS / Radiomaster USB HID)
+ Keyboard/Mouse Input Manager.

Supports:
- Hotplug detection (`JOYDEVICEADDED` / `JOYDEVICEREMOVED`)
- Xbox / DualShock / DualSense / Generic DirectInput pads
- FPV Radio Transmitters (Radiomaster Pocket/Boxer/TX16S, TBS Tango, BetaFPV) in Mode 2
- Customizable deadzones, sensitivity, and live joystick diagnostics screen
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

from .math3d import clamp
from .physics import DroneInput, TruckInput


def apply_deadzone(val: float, dz: float = 0.12) -> float:
    if abs(val) < dz:
        return 0.0
    sign = 1.0 if val > 0.0 else -1.0
    return sign * (abs(val) - dz) / max(1.0 - dz, 1e-4)


# ----------------------------------------------------------------------
# Переназначение клавиш (rebinding)
# ----------------------------------------------------------------------
# Действия, которые можно переназначить. Значения по умолчанию - обычные
# клавиши WASD + мышь, привычные для FPV-пилота.
DEFAULT_KEYBINDS: Dict[str, int] = {
    # --- Оператор FPV-дрона ---
    # ВНИМАНИЕ: у дрона НЕТ ни РЭБ, ни дыма (это умения бензовоза), поэтому
    # таких действий в списке нет - каждая переназначаемая клавиша реально
    # что-то делает.
    "drone_forward": pygame.K_w,
    "drone_back": pygame.K_s,
    "drone_left": pygame.K_a,
    "drone_right": pygame.K_d,
    "drone_throttle_up": pygame.K_LSHIFT,
    "drone_throttle_down": pygame.K_LCTRL,
    "drone_boost": pygame.K_f,
    "drone_detonate": pygame.K_SPACE,
    "drone_acro": pygame.K_x,
    # --- Водитель бензовоза ---
    "truck_accel": pygame.K_w,
    "truck_brake": pygame.K_s,
    "truck_left": pygame.K_a,
    "truck_right": pygame.K_d,
    "truck_fire": pygame.K_SPACE,
    "truck_shotgun": pygame.K_f,
    "truck_ew": pygame.K_e,
    "truck_smoke": pygame.K_q,
    "truck_ready": pygame.K_RETURN,
}

KEYBIND_LABELS_RU: Dict[str, str] = {
    "drone_forward": "FPV: Вперёд (газ по горизонтали)",
    "drone_back": "FPV: Назад",
    "drone_left": "FPV: Влево (крен)",
    "drone_right": "FPV: Вправо (крен)",
    "drone_throttle_up": "FPV: Набор высоты (газ)",
    "drone_throttle_down": "FPV: Снижение высоты",
    "drone_boost": "FPV: Форсаж (boost)",
    "drone_detonate": "FPV: Подрыв БЧ (дистанционный)",
    "drone_acro": "FPV: Режим Acro / Angle",
    "truck_accel": "Бензовоз: Газ",
    "truck_brake": "Бензовоз: Тормоз",
    "truck_left": "Бензовоз: Руль влево",
    "truck_right": "Бензовоз: Руль вправо",
    "truck_fire": "Бензовоз: Огонь из зенитной турели",
    "truck_shotgun": "Бензовоз: Дробовик",
    "truck_ew": "Бензовоз: РЭБ-купол",
    "truck_smoke": "Бензовоз: Дым",
    "truck_ready": "Готовность (подтверждение выбора модификатора)",
}

# Действия, которые можно переназначить (порядок = порядок в меню)
REBINDABLE_ACTIONS: List[str] = list(DEFAULT_KEYBINDS.keys())


def key_name(key_code: int) -> str:
    """Человекочитаемое имя клавиши для отображения в интерфейсе."""
    if key_code is None:
        return "—"
    try:
        name = pygame.key.name(int(key_code))
    except (ValueError, pygame.error):
        name = str(key_code)
    return name.upper() if name else "—"


@dataclass
class JoystickProfile:
    deadzone: float = 0.12
    look_sensitivity: float = 2.6
    mouse_sensitivity: float = 0.0026
    fpv_radio_mode2: bool = False  # True if using Radiomaster/OpenTX/EdgeTX USB HID stick
    # Масштаб внутреннего разрешения мира (1.0 = натуральное, меньше = быстрее)
    render_scale: float = 1.0
    # Переназначенные клавиши: действие -> код pygame
    key_binds: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_KEYBINDS))
    # Калибровка осей: усреднённое значение покоя (центр стика)
    axis_calibration: Dict[int, float] = field(default_factory=dict)

    def save_to(self, path: str) -> bool:
        """Сохраняет профиль в JSON. Возвращает True при успехе."""
        import json
        data = {
            "deadzone": round(float(self.deadzone), 3),
            "look_sensitivity": round(float(self.look_sensitivity), 3),
            "mouse_sensitivity": round(float(self.mouse_sensitivity), 6),
            "fpv_radio_mode2": bool(self.fpv_radio_mode2),
            "render_scale": round(float(self.render_scale), 3),
            "key_binds": {k: int(v) for k, v in self.key_binds.items()},
            "axis_calibration": {str(k): round(float(v), 3) for k, v in self.axis_calibration.items()},
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            return True
        except OSError:
            return False

    @staticmethod
    def load_from(path: str) -> "JoystickProfile":
        """Загружает профиль из JSON; при ошибке - значения по умолчанию."""
        import json
        prof = JoystickProfile()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return prof
        if not isinstance(data, dict):
            return prof
        try:
            prof.deadzone = float(data.get("deadzone", prof.deadzone))
            prof.look_sensitivity = float(data.get("look_sensitivity", prof.look_sensitivity))
            prof.mouse_sensitivity = float(data.get("mouse_sensitivity", prof.mouse_sensitivity))
            prof.fpv_radio_mode2 = bool(data.get("fpv_radio_mode2", prof.fpv_radio_mode2))
            prof.render_scale = max(0.5, min(1.0, float(data.get("render_scale", prof.render_scale))))
        except (TypeError, ValueError):
            pass
        binds = data.get("key_binds")
        if isinstance(binds, dict):
            for action, code in binds.items():
                if action in DEFAULT_KEYBINDS:
                    try:
                        prof.key_binds[action] = int(code)
                    except (TypeError, ValueError):
                        continue
        calib = data.get("axis_calibration")
        if isinstance(calib, dict):
            for axis, val in calib.items():
                try:
                    prof.axis_calibration[int(axis)] = float(val)
                except (TypeError, ValueError):
                    continue
        return prof


@dataclass
class JoystickStatus:
    index: int
    instance_id: int
    name: str
    num_axes: int
    num_buttons: int
    num_hats: int
    axes: List[float] = field(default_factory=list)
    buttons: List[bool] = field(default_factory=list)
    hats: List[Tuple[int, int]] = field(default_factory=list)


class InputController:
    """Unified keyboard + mouse + gamepad/FPV-radio input reader."""

    def __init__(self, profile: Optional[JoystickProfile] = None) -> None:
        self.profile = profile or JoystickProfile()
        self._joysticks: Dict[int, pygame.joystick.JoystickType] = {}
        self.active_joy_id: Optional[int] = None
        # Default to harder Acro (manual rate/attitude hold) for FPV drone; X toggles Angle assist
        self.acro_toggle: bool = True

        # One-shot button/key latch states per frame.
        # Фиксаторы РАЗДЕЛЕНЫ по ролям: иначе клавиши F/E/Q (одинаковые
        # у дрона и бензовоза) срабатывали бы у обеих сторон сразу.
        self._one_detonate = False      # подрыв БЧ (роль: дрон)
        self._one_truck_shotgun = False  # дробовик (роль: бензовоз)
        self._one_truck_ew = False       # РЭБ-купол (роль: бензовоз)
        self._one_truck_smoke = False    # дымовая завеса (роль: бензовоз)
        self._one_ready = False

        self.refresh_joysticks()

    def refresh_joysticks(self) -> None:
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        self._joysticks.clear()
        count = pygame.joystick.get_count()
        for i in range(count):
            try:
                j = pygame.joystick.Joystick(i)
                j.init()
                iid = j.get_instance_id()
                self._joysticks[iid] = j
                # Auto-detect Radiomaster / OpenTX / EdgeTX / BetaFPV USB controllers
                low_name = j.get_name().lower()
                if any(tag in low_name for tag in ("radiomaster", "opentx", "edgetx", "betafpv", "frsky", "tbs")):
                    self.profile.fpv_radio_mode2 = True
            except pygame.error:
                continue
        if self._joysticks and self.active_joy_id not in self._joysticks:
            self.active_joy_id = next(iter(self._joysticks.keys()))
        elif not self._joysticks:
            self.active_joy_id = None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            self.refresh_joysticks()
            return

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_KP_ENTER, pygame.K_r):
                self._one_ready = True
            elif event.key == self.key_for("truck_ready"):
                self._one_ready = True
            elif event.key == self.key_for("drone_detonate"):
                self._one_detonate = True
            elif event.key == self.key_for("truck_shotgun"):
                self._one_truck_shotgun = True
            elif event.key == self.key_for("truck_ew"):
                self._one_truck_ew = True
            elif event.key == self.key_for("truck_smoke"):
                self._one_truck_smoke = True
            elif event.key == self.key_for("drone_boost"):
                # Форсаж - удерживаемое действие, обрабатывается в build_drone_input
                pass
            elif event.key == pygame.K_m:
                self.profile.fpv_radio_mode2 = not self.profile.fpv_radio_mode2
            elif event.key == self.key_for("drone_acro"):
                self.acro_toggle = not self.acro_toggle

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:  # Right Mouse Button -> Shotgun (for Truck) or Detonate (for Drone)
                self._one_truck_shotgun = True
                self._one_detonate = True

        elif event.type == pygame.JOYBUTTONDOWN:
            btn = event.button
            # Standard Xbox/PlayStation mapping:
            # 0=A/Cross (Ready / Detonate), 1=B/Circle (Shotgun), 2=X/Square (EW РЭБ), 3=Y/Triangle (Smoke), 5=RB (Shotgun/Detonate), 7=Start (Ready)
            if btn in (0, 7):
                self._one_ready = True
            if btn in (0, 5):
                self._one_detonate = True
            if btn in (1, 5):
                self._one_truck_shotgun = True
            if btn == 2:
                self._one_truck_ew = True
            if btn == 3:
                self._one_truck_smoke = True
            if btn == 6:
                self.acro_toggle = not self.acro_toggle

    def get_joystick_Snapshot(self) -> List[JoystickStatus]:
        out: List[JoystickStatus] = []
        for idx, (iid, j) in enumerate(self._joysticks.items()):
            try:
                axes = [round(j.get_axis(a), 3) for a in range(j.get_numaxes())]
                btns = [bool(j.get_button(b)) for b in range(j.get_numbuttons())]
                hats = [j.get_hat(h) for h in range(j.get_numhats())]
                out.append(
                    JoystickStatus(
                        index=idx,
                        instance_id=iid,
                        name=j.get_name(),
                        num_axes=j.get_numaxes(),
                        num_buttons=j.get_numbuttons(),
                        num_hats=j.get_numhats(),
                        axes=axes,
                        buttons=btns,
                        hats=hats,
                    )
                )
            except pygame.error:
                continue
        return out

    def _get_active_joy(self) -> Optional[pygame.joystick.JoystickType]:
        if self.active_joy_id is not None and self.active_joy_id in self._joysticks:
            return self._joysticks[self.active_joy_id]
        return None

    # ------------------------------------------------------------------
    # Переназначенные клавиши
    # ------------------------------------------------------------------
    def key_for(self, action: str) -> int:
        """Код клавиши, назначенной на действие (с запасным вариантом)."""
        code = self.profile.key_binds.get(action)
        if code is None:
            code = DEFAULT_KEYBINDS.get(action, 0)
        return int(code)

    def _pressed(self, keys, action: str) -> bool:
        """Нажата ли клавиша, назначенная на действие."""
        code = self.key_for(action)
        if code <= 0:
            return False
        try:
            return bool(keys[code])
        except (IndexError, KeyError):
            return False

    def rebind(self, action: str, key_code: int) -> None:
        """Назначает новую клавишу на действие."""
        if action in DEFAULT_KEYBINDS:
            self.profile.key_binds[action] = int(key_code)

    def reset_binds(self) -> None:
        """Возвращает стандартную раскладку клавиш."""
        self.profile.key_binds = dict(DEFAULT_KEYBINDS)

    def _axis(self, joy: Optional[pygame.joystick.JoystickType], idx: int) -> float:
        """Значение оси с учётом калибровки центра и мёртвой зоны.

        `axis_calibration[idx]` - измеренное значение оси в покое (центр
        стика). Оно вычитается, чтобы стик не «съезжал» при неидеальном
        центрировании; результат ограничивается диапазоном -1..1.
        """
        if joy is None or idx >= joy.get_numaxes():
            return 0.0
        raw = float(joy.get_axis(idx))
        center = float(self.profile.axis_calibration.get(idx, 0.0))
        if center != 0.0:
            raw = clamp(raw - center, -1.0, 1.0)
        return apply_deadzone(raw, self.profile.deadzone)

    def calibrate_active_axes(self) -> int:
        """Калибрует центр осей активного устройства (или всех)."""
        n = 0
        for joy in self._joysticks.values():
            n += self.calibrate_axes(joy)
        return n

    def calibrate_axes(self, joy: Optional[pygame.joystick.JoystickType]) -> int:
        """Запоминает текущие значения осей как «центр» (стик в покое).

        Вызывается, когда игрок отпустил все стики. Возвращает число
        откалиброванных осей.
        """
        if joy is None:
            return 0
        n = joy.get_numaxes()
        for i in range(n):
            self.profile.axis_calibration[i] = round(float(joy.get_axis(i)), 3)
        return n

    def _trigger_axis(self, joy: Optional[pygame.joystick.JoystickType], idx: int) -> float:
        """Converts SDL trigger axis (-1 released .. +1 pressed, or 0..1) into 0..1."""
        if joy is None or idx >= joy.get_numaxes():
            return 0.0
        raw = float(joy.get_axis(idx))
        if raw < -0.05:
            return clamp((raw + 1.0) * 0.5, 0.0, 1.0)
        return clamp(raw, 0.0, 1.0)

    def build_drone_input(self, dt: float, mouse_captured: bool) -> DroneInput:
        keys = pygame.key.get_pressed()
        mx, my = pygame.mouse.get_rel() if mouse_captured else (0, 0)
        mbtns = pygame.mouse.get_pressed(3)
        joy = self._get_active_joy()

        # Клавиши берутся из профиля (их можно переназначить в настройках).
        fwd = (1.0 if self._pressed(keys, "drone_forward") or keys[pygame.K_UP] else 0.0) - (
            1.0 if self._pressed(keys, "drone_back") or keys[pygame.K_DOWN] else 0.0
        )
        strafe = (1.0 if self._pressed(keys, "drone_right") or keys[pygame.K_RIGHT] else 0.0) - (
            1.0 if self._pressed(keys, "drone_left") or keys[pygame.K_LEFT] else 0.0
        )
        throttle = (1.0 if self._pressed(keys, "drone_throttle_up") else 0.0) - (
            1.0 if self._pressed(keys, "drone_throttle_down") else 0.0
        )

        yaw_d = mx * self.profile.mouse_sensitivity
        pitch_d = my * self.profile.mouse_sensitivity
        boost = bool(self._pressed(keys, "drone_boost") or (mouse_captured and mbtns[0]))

        if joy is not None:
            if self.profile.fpv_radio_mode2:
                # FPV Transmitter Mode 2 (EdgeTX / Radiomaster USB):
                # Axis 0 = Roll, Axis 1 = Pitch, Axis 2 = Throttle, Axis 3 = Yaw
                strafe += self._axis(joy, 0)
                fwd += self._axis(joy, 1)
                throttle += self._axis(joy, 2)
                yaw_d += self._axis(joy, 3) * self.profile.look_sensitivity * dt
            else:
                # Standard Gamepad (Xbox / DualSense):
                # Left stick: X=strafe, Y=-forward; Right stick (axes 2/3 or 3/4): Yaw & Pitch
                # LT (axis 4) = descend, RT (axis 5) = climb/boost
                lx = self._axis(joy, 0)
                ly = -self._axis(joy, 1)
                n_axes = joy.get_numaxes()
                if n_axes >= 6:
                    rx = self._axis(joy, 2) if abs(joy.get_axis(2)) < 0.98 or abs(joy.get_axis(3)) > 0.01 else self._axis(joy, 3)
                    # SDL2 GameController standard on Linux/Win: 0=LX,1=LY,2=RX,3=RY,4=LT,5=RT
                    rx = self._axis(joy, 2)
                    ry = self._axis(joy, 3)
                    lt = self._trigger_axis(joy, 4)
                    rt = self._trigger_axis(joy, 5)
                    throttle += rt - lt
                    if joy.get_numbuttons() > 4 and joy.get_button(4):  # LB = Boost
                        boost = True
                else:
                    rx = self._axis(joy, 2)
                    ry = self._axis(joy, 3)

                strafe += lx
                fwd += ly
                yaw_d += rx * self.profile.look_sensitivity * dt
                pitch_d += ry * self.profile.look_sensitivity * dt

        inp = DroneInput(
            throttle=clamp(throttle, -1.0, 1.0),
            forward_Axis=clamp(fwd, -1.0, 1.0),
            strafe_axis=clamp(strafe, -1.0, 1.0),
            yaw_delta=yaw_d,
            pitch_delta=pitch_d,
            boost=boost,
            detonate=self._one_detonate,
            acro_mode=self.acro_toggle,
            ready=self._one_ready,
        )
        self._clear_latches()
        return inp

    def build_truck_input(self, dt: float, mouse_captured: bool) -> TruckInput:
        keys = pygame.key.get_pressed()
        mx, my = pygame.mouse.get_rel() if mouse_captured else (0, 0)
        mbtns = pygame.mouse.get_pressed(3)
        joy = self._get_active_joy()

        accel = (1.0 if self._pressed(keys, "truck_accel") or keys[pygame.K_UP] else 0.0) - (
            1.0 if self._pressed(keys, "truck_brake") or keys[pygame.K_DOWN] else 0.0
        )
        steer = (1.0 if self._pressed(keys, "truck_right") or keys[pygame.K_RIGHT] else 0.0) - (
            1.0 if self._pressed(keys, "truck_left") or keys[pygame.K_LEFT] else 0.0
        )

        aim_yaw_d = mx * self.profile.mouse_sensitivity
        aim_pitch_d = my * self.profile.mouse_sensitivity
        fire_turret = bool(mouse_captured and mbtns[0]) or bool(self._pressed(keys, "truck_fire"))

        if joy is not None:
            steer += self._axis(joy, 0)
            n_axes = joy.get_numaxes()
            if n_axes >= 6:
                rx = self._axis(joy, 2)
                ry = self._axis(joy, 3)
                lt = self._trigger_axis(joy, 4)
                rt = self._trigger_axis(joy, 5)
                # Left stick Y OR triggers can accelerate/brake
                stick_gas = -self._axis(joy, 1)
                accel += stick_gas if abs(stick_gas) > 0.15 else (rt - lt)
                if joy.get_numbuttons() > 4 and joy.get_button(4):  # LB hold = rapid turret
                    fire_turret = True
                if rt > 0.45 and abs(stick_gas) > 0.15:
                    fire_turret = True
            else:
                accel += -self._axis(joy, 1)
                rx = self._axis(joy, 2)
                ry = self._axis(joy, 3)

            aim_yaw_d += rx * self.profile.look_sensitivity * dt
            aim_pitch_d += ry * self.profile.look_sensitivity * dt

        inp = TruckInput(
            accel=clamp(accel, -1.0, 1.0),
            steer=clamp(steer, -1.0, 1.0),
            aim_yaw_delta=aim_yaw_d,
            aim_pitch_delta=aim_pitch_d,
            fire_turret=fire_turret,
            fire_shotgun=self._one_truck_shotgun,
            trigger_ew=self._one_truck_ew,
            trigger_smoke=self._one_truck_smoke,
            ready=self._one_ready,
        )
        self._clear_latches()
        return inp

    def _clear_latches(self) -> None:
        self._one_detonate = False
        self._one_truck_shotgun = False
        self._one_truck_ew = False
        self._one_truck_smoke = False
        self._one_ready = False
