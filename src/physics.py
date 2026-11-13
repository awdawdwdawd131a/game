"""
Authoritative 3D Physics & Match Simulation for Crimean Bridge 1v1:
- Match state machine: MODIFIER_DRAFT (Random Loadout Briefing) -> ACTIVE_ROUND -> ROUND_END -> MATCH_OVER
- Dynamic weather rotation every round (guaranteed distinct consecutive presets)
- Expanded 10+10 random modifiers issued per round via shuffled bags
- Realistic 3D projectile ballistics (gravity drop, aerodynamic drag, truck velocity inheritance,
  continuous swept collision, ballistic lead solver)
- Challenging quadcopter FPV flight dynamics (tilted rotor thrust vectoring, Acro rate-hold default,
  angular inertia, real momentum/gravity, bank-turn coupling, wind & propwash turbulence)
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    BRIDGE_FINISH_Z,
    BRIDGE_START_Z,
    DRONE_BASE_HP,
    DRONE_BASE_LIVES,
    DRONE_BATTERY_SECONDS,
    DRONE_DETONATE_COOLDOWN,
    DRONE_DIRECT_DAMAGE,
    DRONE_DRAG,
    DRONE_EXPLOSION_RADIUS,
    DRONE_GRAVITY,
    DRONE_MAX_SPEED,
    DRONE_MODIFIERS,
    DRONE_RESPAWN_DELAY,
    DRONE_THRUST,
    EW_JAMMER_COOLDOWN,
    EW_JAMMER_DURATION,
    EW_JAMMER_RADIUS,
    MAX_ROUNDS_TO_WIN,
    MODIFIER_PICK_TIME,
    ROAD_DECK_Y,
    ROAD_HALF_WIDTH,
    ROUND_END_DELAY,
    ROUND_TIME_LIMIT,
    SHOTGUN_COOLDOWN,
    SHOTGUN_PELLET_DMG,
    SHOTGUN_PELLET_DRAG,
    SHOTGUN_PELLET_SPEED,
    SHOTGUN_PELLETS,
    SHOTGUN_RANGE,
    SHOTGUN_SPREAD_DEG,
    SMOKE_COOLDOWN,
    SMOKE_DURATION,
    SMOKE_RADIUS,
    TRUCK_BASE_ACCEL,
    TRUCK_BASE_BRAKE,
    TRUCK_BASE_HP,
    TRUCK_BASE_MAX_SPEED,
    TRUCK_MODIFIERS,
    TRUCK_STEER_RATE,
    TURRET_BULLET_DRAG,
    TURRET_BULLET_SPEED,
    TURRET_COOL_RATE,
    TURRET_DMG,
    TURRET_FIRE_INTERVAL,
    TURRET_HEAT_PER_SHOT,
    TURRET_LOCK_CONE,
    TURRET_LOCK_TIME,
    TURRET_RANGE,
    WEATHER_BY_ID,
    WEATHER_PRESETS,
    WeatherPreset,
)
from .math3d import (
    AABB,
    clamp,
    euler_to_matrix,
    forward_from_angles,
    generate_barrier_layout,
    move_toward,
    segment_sphere_intersect,
    wrap_angle,
)


@dataclass
class DroneInput:
    throttle: float = 0.0        # -1..+1 (collective motor throttle)
    forward_Axis: float = 0.0    # -1..+1 cyclic pitch forward/back
    strafe_axis: float = 0.0     # -1..+1 roll left/right (tilts thrust vector!)
    yaw_delta: float = 0.0       # radians turn delta this tick (mouse/stick)
    pitch_delta: float = 0.0     # radians look pitch delta this tick
    boost: bool = False          # sprint / nitro motors
    detonate: bool = False       # manual airburst / cluster drop trigger
    acro_mode: bool = True       # True = Acro (manual rate hold), False = Angle (self-level assist)
    pick_modifier: str = ""      # deprecated / unused in random-issue mode (kept for wire compatibility)
    ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DroneInput":
        valid = {k: v for k, v in d.items() if k in DroneInput.__dataclass_fields__}
        return DroneInput(**valid)


@dataclass
class TruckInput:
    accel: float = 0.0           # -1 (brake/reverse) .. +1 (full throttle)
    steer: float = 0.0           # -1 (left) .. +1 (right)
    aim_yaw_delta: float = 0.0   # turret/gunner yaw delta (radians)
    aim_pitch_delta: float = 0.0 # turret/gunner pitch delta (radians)
    fire_turret: bool = False    # LMB / RT hold rapid AA turret
    fire_shotgun: bool = False   # RMB / RB anti-drone buckshot blast
    trigger_ew: bool = False     # E / X button: activate EW Jammer (РЭБ)
    trigger_smoke: bool = False  # Q / Y button: deploy smoke screen
    pick_modifier: str = ""      # deprecated / unused in random-issue mode (kept for wire compatibility)
    ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TruckInput":
        valid = {k: v for k, v in d.items() if k in TruckInput.__dataclass_fields__}
        return TruckInput(**valid)


@dataclass
class Projectile:
    pos: np.ndarray
    prev_pos: np.ndarray
    vel: np.ndarray
    damage: float
    hit_radius: float
    drag: float
    gravity_mult: float
    ttl: float
    initial_speed: float
    color: Tuple[int, int, int]
    weapon_name: str
    kind: str = "turret"  # "turret", "flak", "pellet", "bomblet"


@dataclass
class TracerFX:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    color: Tuple[int, int, int]
    ttl: float
    kind: str = "turret"


@dataclass
class ExplosionFX:
    pos: Tuple[float, float, float]
    radius: float
    ttl: float
    max_ttl: float


@dataclass
class SmokeCloud:
    pos: Tuple[float, float, float]
    radius: float
    ttl: float


class WorldSimulation:
    """
    Deterministic/authoritative 3D match & round simulation.
    Used identically by:
    - Dedicated / Hosted Multiplayer Server
    - Local Practice vs Smart Bot
    - Automated Self-Test Verification Suite
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.barriers: List[AABB] = generate_barrier_layout(seed=42)
        self.static_colliders: List[AABB] = self._build_colliders()

        # Match progression
        self.phase: str = "MODIFIER_DRAFT"   # MODIFIER_DRAFT, ACTIVE_ROUND, ROUND_END, MATCH_OVER
        self.round_number: int = 1
        self.drone_score: int = 0
        self.truck_score: int = 0
        self.phase_timer: float = MODIFIER_PICK_TIME
        self.round_winner: str = ""          # "drone" or "truck"
        self.round_reason_ru: str = ""
        self.match_winner: str = ""

        # Shuffled bags for random non-repeating Weather and Modifiers!
        self._weather_bag: List[str] = []
        self._drone_mod_bag: List[str] = []
        self._truck_mod_bag: List[str] = []
        self.weather_id: str = "clear_day"
        self.lightning_flash: float = 0.0
        self.sim_time: float = 0.0

        # Active Modifiers per player (randomly issued each round)
        self.drone_mod: str = DRONE_MODIFIERS[0].id
        self.truck_mod: str = TRUCK_MODIFIERS[0].id
        self.drone_draft_ready: bool = False
        self.truck_draft_ready: bool = False

        # Fuel Truck State
        self.truck_pos = np.array([3.2, ROAD_DECK_Y, BRIDGE_START_Z], dtype=np.float32)
        self.truck_vel = np.zeros(3, dtype=np.float32)
        self.truck_speed: float = 0.0
        self.truck_yaw: float = 0.0
        self.truck_hp: float = TRUCK_BASE_HP
        self.truck_max_hp: float = TRUCK_BASE_HP
        self.truck_reactive_ready: bool = False
        self.truck_last_hit_timer: float = 99.0
        self.turret_yaw: float = 0.0
        self.turret_pitch: float = -0.18
        self.turret_heat: float = 0.0
        self.turret_lock: float = 0.0
        self.turret_overheated: bool = False
        self.turret_cd: float = 0.0
        self.shotgun_cd: float = 0.0
        self.ew_active_timer: float = 0.0
        self.ew_cd: float = 0.0
        self.smoke_cd: float = 0.0

        # FPV Drone State
        self.drone_pos = np.array([0.0, ROAD_DECK_Y + 26.0, BRIDGE_START_Z + 420.0], dtype=np.float32)
        self.drone_vel = np.zeros(3, dtype=np.float32)
        self.drone_yaw: float = math.pi
        self.drone_pitch: float = 0.08
        self.drone_roll: float = 0.0
        self.drone_yaw_rate: float = 0.0
        self.drone_pitch_rate: float = 0.0
        self.drone_roll_rate: float = 0.0
        self.drone_throttle_pct: float = 35.0
        self.drone_hp: float = DRONE_BASE_HP
        self.drone_max_hp: float = DRONE_BASE_HP
        self.drone_battery: float = DRONE_BATTERY_SECONDS
        self.drone_max_battery: float = DRONE_BATTERY_SECONDS
        self.drone_lives_left: int = DRONE_BASE_LIVES
        self.drone_alive: bool = True
        self.drone_respawn_timer: float = 0.0
        self.drone_jam_level: float = 0.0
        self.drone_detonate_cd: float = 0.0
        self.drone_acro: bool = True

        # Active ballistic projectiles & FX synced to clients
        self.projectiles: List[Projectile] = []
        self.net_proj_lines: List[TracerFX] = []
        self.tracers: List[TracerFX] = []
        self.explosions: List[ExplosionFX] = []
        self.smokes: List[SmokeCloud] = []
        self.last_event_ru: str = "Раунд 1: случайные модификаторы выданы!"

        self._pick_next_weather()
        self._pick_next_modifiers()
        self._init_round_entities()

    def _build_colliders(self) -> List[AABB]:
        cols = list(self.barriers)
        arch_steps = 18
        for s in range(arch_steps):
            u0 = -1.0 + 2.0 * (s / arch_steps)
            u1 = -1.0 + 2.0 * ((s + 1) / arch_steps)
            z0 = u0 * 115.0
            z1 = u1 * 115.0
            y0 = ROAD_DECK_Y + (63.0 - ROAD_DECK_Y) * (1.0 - u0 * u0)
            y1 = ROAD_DECK_Y + (63.0 - ROAD_DECK_Y) * (1.0 - u1 * u1)
            for side_x in (-ROAD_HALF_WIDTH - 0.9, ROAD_HALF_WIDTH + 0.9):
                cols.append(
                    AABB(
                        min_pt=np.array([side_x - 1.2, min(y0, y1) - 1.2, min(z0, z1) - 1.2], dtype=np.float32),
                        max_pt=np.array([side_x + 1.2, max(y0, y1) + 1.2, max(z0, z1) + 1.2], dtype=np.float32),
                        tag="arch",
                    )
                )
        return cols

    @property
    def weather(self) -> WeatherPreset:
        return WEATHER_BY_ID[self.weather_id]

    def _pick_next_weather(self) -> None:
        if not self._weather_bag:
            ids = [w.id for w in WEATHER_PRESETS]
            self.rng.shuffle(ids)
            if ids[0] == self.weather_id and len(ids) > 1:
                ids[0], ids[-1] = ids[-1], ids[0]
            self._weather_bag = ids
        self.weather_id = self._weather_bag.pop(0)

    def _pick_next_modifiers(self) -> None:
        """Draws a fresh random modifier for both FPV Drone and Fuel Truck without repeats."""
        if not self._drone_mod_bag:
            ids = [m.id for m in DRONE_MODIFIERS]
            self.rng.shuffle(ids)
            if ids[0] == self.drone_mod and len(ids) > 1:
                ids[0], ids[-1] = ids[-1], ids[0]
            self._drone_mod_bag = ids
        self.drone_mod = self._drone_mod_bag.pop(0)

        if not self._truck_mod_bag:
            ids = [m.id for m in TRUCK_MODIFIERS]
            self.rng.shuffle(ids)
            if ids[0] == self.truck_mod and len(ids) > 1:
                ids[0], ids[-1] = ids[-1], ids[0]
            self._truck_mod_bag = ids
        self.truck_mod = self._truck_mod_bag.pop(0)

    def reset_full_match(self) -> None:
        self.round_number = 1
        self.drone_score = 0
        self.truck_score = 0
        self.match_winner = ""
        self._weather_bag.clear()
        self._drone_mod_bag.clear()
        self._truck_mod_bag.clear()
        self.start_modifier_draft(advance_weather=True)

    def start_modifier_draft(self, advance_weather: bool = True) -> None:
        if advance_weather:
            self._pick_next_weather()
        self._pick_next_modifiers()
        self.phase = "MODIFIER_DRAFT"
        self.phase_timer = MODIFIER_PICK_TIME
        self.drone_draft_ready = False
        self.truck_draft_ready = False
        self.round_winner = ""
        self.round_reason_ru = ""
        self._init_round_entities()
        self.last_event_ru = f"Раунд {self.round_number} | Погода: {self.weather.name_ru} | Случайные моды выданы!"

    def _init_round_entities(self) -> None:
        self.projectiles.clear()
        self.net_proj_lines.clear()
        self.tracers.clear()
        self.explosions.clear()
        self.smokes.clear()

        # Apply Truck Modifiers
        bonus_hp = 0.0
        if self.truck_mod == "anti_drone_cage":
            bonus_hp = 20.0
        elif self.truck_mod == "reactive_armor":
            bonus_hp = 15.0
        elif self.truck_mod == "auto_extinguisher":
            bonus_hp = 35.0
        self.truck_max_hp = TRUCK_BASE_HP + bonus_hp
        self.truck_hp = self.truck_max_hp
        self.truck_reactive_ready = self.truck_mod == "reactive_armor"
        self.truck_last_hit_timer = 99.0

        self.truck_pos = np.array([3.2, ROAD_DECK_Y, BRIDGE_START_Z], dtype=np.float32)
        self.truck_vel = np.zeros(3, dtype=np.float32)
        self.truck_speed = 14.0  # Rolling start onto the bridge
        self.truck_yaw = 0.0
        self.turret_yaw = 0.0    # Facing forward (+Z) toward the arches & incoming FPV!
        self.turret_pitch = -0.18
        self.turret_heat = 0.0
        self.turret_lock = 0.0
        self.turret_overheated = False
        self.turret_cd = 0.0
        self.shotgun_cd = 0.0
        self.ew_active_timer = 0.0
        self.ew_cd = 0.0
        self.smoke_cd = 0.0

        # Apply Drone Modifiers
        extra_lives = 1 if self.drone_mod == "kevlar_frame" else 0
        self.drone_lives_left = DRONE_BASE_LIVES + extra_lives
        self._spawn_fresh_drone(initial=True)

    def _spawn_fresh_drone(self, initial: bool = False) -> None:
        hp_mult = 1.0
        if self.drone_mod == "kevlar_frame":
            hp_mult = 1.45
        elif self.drone_mod == "long_range_lipo":
            hp_mult = 1.25

        bat_mult = 1.0
        if self.drone_mod == "fiber_optic":
            bat_mult = 1.20
        elif self.drone_mod == "long_range_lipo":
            bat_mult = 1.75

        self.drone_max_hp = DRONE_BASE_HP * hp_mult
        self.drone_hp = self.drone_max_hp
        self.drone_max_battery = DRONE_BATTERY_SECONDS * bat_mult
        self.drone_battery = self.drone_max_battery
        self.drone_alive = True
        self.drone_respawn_timer = 0.0
        self.drone_jam_level = 0.0
        self.drone_detonate_cd = 0.0

        # Spawn just beyond the iconic white Crimean Bridge arches facing the oncoming truck
        spawn_z = clamp(float(self.truck_pos[2]) + 330.0, 95.0, BRIDGE_FINISH_Z + 40.0)
        spawn_x = float(self.rng.choice([-8.5, 0.0, 8.5, 19.0]))
        spawn_y = ROAD_DECK_Y + (28.0 if initial else 34.0)
        self.drone_pos = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float32)
        self.drone_vel = np.array([0.0, 0.0, -14.0], dtype=np.float32)
        dx = float(self.truck_pos[0] - self.drone_pos[0])
        dz = float(self.truck_pos[2] - self.drone_pos[2])
        self.drone_yaw = math.atan2(dx, dz)
        self.drone_pitch = 0.08
        self.drone_roll = 0.0
        self.drone_yaw_rate = 0.0
        self.drone_pitch_rate = 0.0
        self.drone_roll_rate = 0.0
        self.drone_throttle_pct = 35.0
        self.drone_acro = True

    def begin_active_round(self) -> None:
        self.phase = "ACTIVE_ROUND"
        self.phase_timer = ROUND_TIME_LIMIT
        self._init_round_entities()
        self.last_event_ru = f"БОЙ НАЧАЛСЯ! Погода: {self.weather.name_ru}"

    # -----------------------------------------------------------------------
    # Ballistic Lead Prediction Solver
    # -----------------------------------------------------------------------
    def get_turret_bullet_speed(self) -> float:
        spd = TURRET_BULLET_SPEED
        if self.truck_mod == "twin_zu_turret":
            spd *= 1.15
        elif self.truck_mod == "ballistic_computer":
            spd *= 1.25
        return spd

    def compute_ballistic_lead_pos(
        self,
        bullet_speed: Optional[float] = None,
        gravity_mult: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Iterative 2-pass ballistic intercept point accounting for bullet speed,
        relative drone velocity, truck velocity inheritance, and gravity drop.
        """
        if not self.drone_alive:
            return None
        spd = bullet_speed if bullet_speed is not None else self.get_turret_bullet_speed()
        muzzle = self.get_turret_muzzle_pos()
        rel_vel = self.drone_vel - self.truck_vel
        rel_pos = self.drone_pos - muzzle
        dist = float(np.linalg.norm(rel_pos))
        t = dist / max(spd, 1.0)
        pred = self.drone_pos + rel_vel * t + np.array([0.0, 0.5 * DRONE_GRAVITY * gravity_mult * t * t, 0.0], dtype=np.float32)
        t = float(np.linalg.norm(pred - muzzle)) / max(spd, 1.0)
        pred = self.drone_pos + rel_vel * t + np.array([0.0, 0.5 * DRONE_GRAVITY * gravity_mult * t * t, 0.0], dtype=np.float32)
        return pred

    # -----------------------------------------------------------------------
    # Main Simulation Step
    # -----------------------------------------------------------------------
    def step(self, dt: float, drone_in: DroneInput, truck_in: TruckInput) -> None:
        self.sim_time += dt
        self._tick_fx(dt)

        if self.lightning_flash > 0.0:
            self.lightning_flash = max(0.0, self.lightning_flash - dt * 3.2)
        elif self.weather.lightning_chance > 0.0:
            if self.rng.random() < self.weather.lightning_chance * dt:
                self.lightning_flash = 1.0

        if self.phase == "MODIFIER_DRAFT":
            # Modifiers are issued randomly per round; players confirm readiness
            if drone_in.ready:
                self.drone_draft_ready = True
            if truck_in.ready:
                self.truck_draft_ready = True

            self.phase_timer -= dt
            if (self.drone_draft_ready and self.truck_draft_ready) or self.phase_timer <= 0.0:
                self.begin_active_round()
            return

        if self.phase == "ROUND_END":
            self.phase_timer -= dt
            if self.phase_timer <= 0.0:
                if self.drone_score >= MAX_ROUNDS_TO_WIN or self.truck_score >= MAX_ROUNDS_TO_WIN:
                    self.phase = "MATCH_OVER"
                    self.match_winner = "drone" if self.drone_score > self.truck_score else "truck"
                else:
                    self.round_number += 1
                    self.start_modifier_draft(advance_weather=True)
            return

        if self.phase == "MATCH_OVER":
            if drone_in.ready and truck_in.ready:
                self.reset_full_match()
            return

        # --- ACTIVE_ROUND ---
        self.phase_timer -= dt
        prev_drone_pos = self.drone_pos.copy()
        prev_drone_alive = self.drone_alive
        self._step_truck(dt, truck_in)
        self._step_drone(dt, drone_in)
        self._step_projectiles(dt, prev_drone_pos, prev_drone_alive)
        self._check_round_conditions()

    def _tick_fx(self, dt: float) -> None:
        self.tracers = [t for t in self.tracers if (setattr(t, "ttl", t.ttl - dt) or t.ttl > 0.0)]
        self.explosions = [e for e in self.explosions if (setattr(e, "ttl", e.ttl - dt) or e.ttl > 0.0)]
        self.smokes = [s for s in self.smokes if (setattr(s, "ttl", s.ttl - dt) or s.ttl > 0.0)]

    # -----------------------------------------------------------------------
    # Truck Driving & Anti-FPV Ballistic Weapons
    # -----------------------------------------------------------------------
    def _step_truck(self, dt: float, inp: TruckInput) -> None:
        w = self.weather
        v12 = self.truck_mod == "v12_biturbo"
        max_spd = TRUCK_BASE_MAX_SPEED * (1.25 if v12 else 1.0)
        accel_rate = TRUCK_BASE_ACCEL * (1.35 if v12 else 1.0)
        grip = min(1.0, w.truck_grip * (1.40 if v12 else 1.0))

        # Auto-extinguisher passive repair after 4.0s out of combat blast
        self.truck_last_hit_timer += dt
        if self.truck_mod == "auto_extinguisher" and self.truck_last_hit_timer >= 4.0 and self.truck_hp > 0.0:
            self.truck_hp = min(self.truck_max_hp, self.truck_hp + 2.5 * dt)

        # Longitudinal acceleration / braking
        if inp.accel >= 0.0:
            self.truck_speed += inp.accel * accel_rate * grip * dt
        else:
            if self.truck_speed > 0.5:
                self.truck_speed += inp.accel * TRUCK_BASE_BRAKE * grip * dt
            else:
                self.truck_speed += inp.accel * (accel_rate * 0.55) * grip * dt

        self.truck_speed -= self.truck_speed * 0.12 * dt
        self.truck_speed = clamp(self.truck_speed, -9.0, max_spd)

        speed_factor = clamp(abs(self.truck_speed) / 8.0, 0.28, 1.0)
        reverse_sign = -1.0 if self.truck_speed < -0.2 else 1.0
        max_yaw_from_bridge = 0.62
        self.truck_yaw += inp.steer * TRUCK_STEER_RATE * grip * speed_factor * reverse_sign * dt
        self.truck_yaw = clamp(self.truck_yaw, -max_yaw_from_bridge, max_yaw_from_bridge)

        vx = math.sin(self.truck_yaw) * self.truck_speed
        vz = math.cos(self.truck_yaw) * self.truck_speed
        self.truck_vel[:] = (vx, 0.0, vz)
        self.truck_pos[0] += vx * dt
        self.truck_pos[2] += vz * dt
        self.truck_pos[1] = ROAD_DECK_Y

        guard_limit = ROAD_HALF_WIDTH - 2.1
        if abs(float(self.truck_pos[0])) > guard_limit:
            self.truck_pos[0] = clamp(float(self.truck_pos[0]), -guard_limit, guard_limit)
            self.truck_speed *= 0.90
            self.truck_yaw *= 0.7

        self.truck_pos[2] = max(BRIDGE_START_Z - 20.0, float(self.truck_pos[2]))

        tx, tz = float(self.truck_pos[0]), float(self.truck_pos[2])
        half_tw, half_tl = 1.55, 5.6
        for b in self.barriers:
            if (tx + half_tw > b.min_pt[0] and tx - half_tw < b.max_pt[0] and
                    tz + half_tl > b.min_pt[2] and tz - half_tl < b.max_pt[2]):
                bcx = 0.5 * float(b.min_pt[0] + b.max_pt[0])
                push_dir = 1.0 if tx >= bcx else -1.0
                self.truck_pos[0] = bcx + push_dir * (0.5 * float(b.max_pt[0] - b.min_pt[0]) + half_tw + 0.15)
                self.truck_speed *= 0.70
                self.last_event_ru = "Бензовоз задел бетонный блок!"

        # Gunner / Turret Aiming (allows near-vertical overhead elevation -1.48 rad!)
        self.turret_yaw = wrap_angle(self.turret_yaw + inp.aim_yaw_delta)
        self.turret_pitch = clamp(self.turret_pitch + inp.aim_pitch_delta, -1.48, 0.28)

        # Cooldowns
        self.turret_cd = max(0.0, self.turret_cd - dt)
        self.shotgun_cd = max(0.0, self.shotgun_cd - dt)
        self.ew_cd = max(0.0, self.ew_cd - dt)
        self.smoke_cd = max(0.0, self.smoke_cd - dt)
        if self.ew_active_timer > 0.0:
            self.ew_active_timer = max(0.0, self.ew_active_timer - dt)

        cool_mult = 1.4 if self.truck_mod == "twin_zu_turret" else 1.0
        self.turret_heat = max(0.0, self.turret_heat - TURRET_COOL_RATE * cool_mult * dt)
        if self.turret_overheated and self.turret_heat < 0.25:
            self.turret_overheated = False

        muzzle_pos = self.get_turret_muzzle_pos()
        aim_dir = forward_from_angles(self.turret_yaw, self.turret_pitch)

        # Ballistic FCS lock meter (purely visual tracking lock indicator — NO damage penalty!)
        if self.drone_alive:
            lead_pt = self.compute_ballistic_lead_pos()
            track_target = lead_pt if lead_pt is not None else self.drone_pos
            to_target = track_target - muzzle_pos
            trange = float(np.linalg.norm(to_target))
            if trange > 1e-3:
                ang_err = math.acos(clamp(float(np.dot(aim_dir, to_target / trange)), -1.0, 1.0))
                if ang_err < TURRET_LOCK_CONE and trange <= TURRET_RANGE:
                    self.turret_lock = min(1.0, self.turret_lock + dt / TURRET_LOCK_TIME)
                else:
                    self.turret_lock = max(0.0, self.turret_lock - dt * 1.6)
        else:
            self.turret_lock = max(0.0, self.turret_lock - dt * 1.6)

        # 1) Rapid Roof AA Turret (Physical 3D Ballistic Projectiles!)
        if inp.fire_turret and not self.turret_overheated and self.turret_cd <= 0.0:
            twin = self.truck_mod == "twin_zu_turret"
            bcomp = self.truck_mod == "ballistic_computer"
            flak = self.truck_mod == "flak_shrapnel"
            self.turret_cd = TURRET_FIRE_INTERVAL * (0.58 if twin else 1.0)
            self.turret_heat += TURRET_HEAT_PER_SHOT * (0.60 if twin else 1.0)
            if self.turret_heat >= 1.0:
                self.turret_heat = 1.0
                self.turret_overheated = True
                self.last_event_ru = "Перегрев зенитной турели!"

            base_spread = 1.05 if twin else 1.25
            if bcomp:
                base_spread *= 0.75
            shot_dir = self._jitter_dir(aim_dir, math.radians(base_spread))
            muzzle_spd = self.get_turret_bullet_speed()
            vel = shot_dir * muzzle_spd + self.truck_vel

            hit_rad = 0.95 * (1.55 if flak else 1.0)
            if self.drone_mod == "stealth_matte":
                hit_rad *= 0.72

            dmg = TURRET_DMG * (1.08 if twin else 1.0)
            col = (255, 195, 65) if flak else (255, 225, 85)
            kind = "flak" if flak else "turret"
            ttl = TURRET_RANGE / max(muzzle_spd * 0.85, 1.0) + 0.15

            self.projectiles.append(
                Projectile(
                    pos=muzzle_pos.copy(),
                    prev_pos=muzzle_pos.copy(),
                    vel=vel.astype(np.float32),
                    damage=dmg,
                    hit_radius=hit_rad,
                    drag=TURRET_BULLET_DRAG,
                    gravity_mult=1.0,
                    ttl=ttl,
                    initial_speed=muzzle_spd,
                    color=col,
                    weapon_name="Зенитный фугас" if flak else "ЗПУ турель",
                    kind=kind,
                )
            )
            # Short muzzle flash tracer (also drives gunshot audio cue)
            flash_end = muzzle_pos + shot_dir * 4.5
            self.tracers.append(
                TracerFX(
                    start=(float(muzzle_pos[0]), float(muzzle_pos[1]), float(muzzle_pos[2])),
                    end=(float(flash_end[0]), float(flash_end[1]), float(flash_end[2])),
                    color=col,
                    ttl=0.045,
                )
            )

        # 2) Anti-FPV Buckshot Shotgun ("Перехват" — Ballistic Pellet Cloud!)
        if inp.fire_shotgun and self.shotgun_cd <= 0.0:
            buck_mod = self.truck_mod == "buckshot_radar"
            self.shotgun_cd = SHOTGUN_COOLDOWN * (0.82 if buck_mod else 1.0)
            pellets = SHOTGUN_PELLETS * (2 if buck_mod else 1)
            spread_rad = math.radians(SHOTGUN_SPREAD_DEG * (1.12 if buck_mod else 1.0))
            max_rng = SHOTGUN_RANGE * (1.15 if buck_mod else 1.0)
            pellet_spd = SHOTGUN_PELLET_SPEED * (1.08 if buck_mod else 1.0)
            hit_rad = 1.15 * (0.72 if self.drone_mod == "stealth_matte" else 1.0)
            ttl = max_rng / max(pellet_spd * 0.72, 1.0)

            for i in range(pellets):
                pdir = self._jitter_dir(aim_dir, spread_rad)
                pvel = pdir * pellet_spd + self.truck_vel
                self.projectiles.append(
                    Projectile(
                        pos=muzzle_pos.copy(),
                        prev_pos=muzzle_pos.copy(),
                        vel=pvel.astype(np.float32),
                        damage=SHOTGUN_PELLET_DMG,
                        hit_radius=hit_rad,
                        drag=SHOTGUN_PELLET_DRAG,
                        gravity_mult=1.0,
                        ttl=ttl,
                        initial_speed=pellet_spd,
                        color=(255, 155, 60),
                        weapon_name="Дробовик 12к",
                        kind="pellet",
                    )
                )
                if i < 6:
                    f_end = muzzle_pos + pdir * 4.0
                    self.tracers.append(
                        TracerFX(
                            start=(float(muzzle_pos[0]), float(muzzle_pos[1]), float(muzzle_pos[2])),
                            end=(float(f_end[0]), float(f_end[1]), float(f_end[2])),
                            color=(255, 165, 70),
                            ttl=0.055,
                        )
                    )

        # 3) EW Jammer ("РЭБ Купол")
        if inp.trigger_ew and self.ew_cd <= 0.0:
            kupol = self.truck_mod == "ew_kupol"
            tda = self.truck_mod == "flamethrower_smoke"
            cd_mult = 0.70 if kupol else (0.75 if tda else 1.0)
            self.ew_active_timer = EW_JAMMER_DURATION * (1.35 if kupol else 1.0)
            self.ew_cd = EW_JAMMER_COOLDOWN * cd_mult
            self.last_event_ru = "Водитель включил купол РЭБ!"

        # 4) Smoke Screen ("Аэрозольная завеса" / ТДА)
        if inp.trigger_smoke and self.smoke_cd <= 0.0:
            buck_mod = self.truck_mod == "buckshot_radar"
            tda = self.truck_mod == "flamethrower_smoke"
            cd_mult = 0.55 if buck_mod else (0.70 if tda else 1.0)
            self.smoke_cd = SMOKE_COOLDOWN * cd_mult
            rad = SMOKE_RADIUS * (1.55 if tda else 1.0)
            dur = 9.0 if tda else SMOKE_DURATION
            rear_pos = self.truck_pos - np.array([math.sin(self.truck_yaw) * 4.0, -2.2, math.cos(self.truck_yaw) * 4.0], dtype=np.float32)
            self.smokes.append(
                SmokeCloud(
                    pos=(float(rear_pos[0]), float(rear_pos[1]), float(rear_pos[2])),
                    radius=rad,
                    ttl=dur,
                )
            )
            if tda:
                front_pos = self.truck_pos + np.array([math.sin(self.truck_yaw) * 8.0, 2.5, math.cos(self.truck_yaw) * 8.0], dtype=np.float32)
                self.smokes.append(
                    SmokeCloud(
                        pos=(float(front_pos[0]), float(front_pos[1]), float(front_pos[2])),
                        radius=rad * 0.9,
                        ttl=dur,
                    )
                )
            self.last_event_ru = "Поставлена термодымовая завеса ТДА!" if tda else "Поставлена дымовая завеса!"

    def _step_projectiles(self, dt: float, prev_drone_pos: np.ndarray, prev_drone_alive: bool) -> None:
        """Simulates physical 3D projectile ballistics: gravity, air drag, swept drone & world hits."""
        if not self.projectiles:
            return
        w = self.weather
        wind = np.array(w.wind_vec, dtype=np.float32)
        survivors: List[Projectile] = []

        for p in self.projectiles:
            p.ttl -= dt
            if p.ttl <= 0.0:
                continue

            p.prev_pos = p.pos.copy()
            # Gravity + aerodynamic drag + subtle wind drift
            p.vel[1] -= DRONE_GRAVITY * p.gravity_mult * dt
            p.vel -= p.vel * p.drag * dt
            if p.kind != "bomblet":
                p.vel += wind * 0.05 * dt
            p.pos = p.pos + p.vel * dt

            # Cluster bomblet impact on truck or bridge deck
            if p.kind == "bomblet":
                hit_truck = False
                truck_center = self.truck_pos + np.array([0.0, 2.15, -0.3], dtype=np.float32)
                rel = p.pos - truck_center
                cy, sy = math.cos(-self.truck_yaw), math.sin(-self.truck_yaw)
                lx = rel[0] * cy + rel[2] * sy
                ly = rel[1]
                lz = -rel[0] * sy + rel[2] * cy
                if abs(lx) <= 2.3 and abs(ly) <= 2.4 and abs(lz) <= 6.2:
                    hit_truck = True
                hit_deck = (
                    abs(float(p.pos[0])) <= ROAD_HALF_WIDTH + 0.5
                    and float(p.pos[1]) <= ROAD_DECK_Y + 0.15
                )
                hit_sea = float(p.pos[1]) <= 0.2
                if hit_truck or hit_deck or hit_sea:
                    self._explode_point(p.pos, radius=10.5, base_dmg=p.damage, reason_ru="Кассетный суббоеприпас")
                    continue
                survivors.append(p)
                continue

            # Anti-drone bullets & pellets: continuous swept collision against drone's motion this tick!
            if self.drone_alive and prev_drone_alive:
                # Relative segment in drone rest frame -> prevents tunneling at 450 m/s!
                rel_start = p.prev_pos - prev_drone_pos
                rel_end = p.pos - self.drone_pos
                hit_t = segment_sphere_intersect(rel_start, rel_end, np.zeros(3, dtype=np.float32), p.hit_radius)
                if hit_t is not None:
                    spd_now = float(np.linalg.norm(p.vel))
                    vel_ratio = clamp(spd_now / max(p.initial_speed, 1.0), 0.40, 1.05)
                    dealt = p.damage * (vel_ratio if p.kind == "pellet" else (0.70 + 0.30 * vel_ratio))
                    self.drone_hp -= dealt
                    hit_pt = p.prev_pos + (p.pos - p.prev_pos) * hit_t
                    bdir = p.vel / max(spd_now, 1e-4)
                    self.drone_vel += bdir * (dealt * 0.28)
                    if p.kind == "flak":
                        self.explosions.append(
                            ExplosionFX(
                                pos=(float(hit_pt[0]), float(hit_pt[1]), float(hit_pt[2])),
                                radius=2.6,
                                ttl=0.22,
                                max_ttl=0.22,
                            )
                        )
                    if self.drone_hp <= 0.0:
                        self._destroy_drone_in_air(f"СБИТ FPV ДРОН ({p.weapon_name})!")
                    continue

            # World stop (sea or roadway)
            if float(p.pos[1]) <= 0.0 or (abs(float(p.pos[0])) <= ROAD_HALF_WIDTH and float(p.pos[1]) <= ROAD_DECK_Y and abs(float(p.pos[2])) <= 750.0):
                continue

            survivors.append(p)

        self.projectiles = survivors

    def get_turret_muzzle_pos(self) -> np.ndarray:
        cx = float(self.truck_pos[0]) + math.sin(self.truck_yaw) * 3.1
        cy = float(self.truck_pos[1]) + 3.65
        cz = float(self.truck_pos[2]) + math.cos(self.truck_yaw) * 3.1
        return np.array([cx, cy, cz], dtype=np.float32)

    def _jitter_dir(self, base_dir: np.ndarray, max_angle_rad: float) -> np.ndarray:
        jx = self.rng.gauss(0.0, max_angle_rad * 0.55)
        jy = self.rng.gauss(0.0, max_angle_rad * 0.55)
        jz = self.rng.gauss(0.0, max_angle_rad * 0.55)
        v = base_dir + np.array([jx, jy, jz], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-6)

    # -----------------------------------------------------------------------
    # FPV Drone Flight (Harder Quadcopter Physics), Wind, EW & Detonation
    # -----------------------------------------------------------------------
    def _step_drone(self, dt: float, inp: DroneInput) -> None:
        if not self.drone_alive:
            self.drone_respawn_timer -= dt
            if self.drone_respawn_timer <= 0.0 and self.drone_lives_left > 0:
                self._spawn_fresh_drone(initial=False)
                self.last_event_ru = f"Оператор поднял новый FPV (осталось: {self.drone_lives_left})"
            return

        w = self.weather
        sport = self.drone_mod == "sport_motors"
        heavy_pg = self.drone_mod == "shaped_charge"
        fiber = self.drone_mod == "fiber_optic"
        wind_gyro = self.drone_mod == "wind_gyro"
        lipo = self.drone_mod == "long_range_lipo"

        # EW Jammer calculation
        dist_to_truck = float(np.linalg.norm(self.drone_pos - (self.truck_pos + np.array([0.0, 2.2, 0.0], dtype=np.float32))))
        ew_rad = EW_JAMMER_RADIUS * (1.60 if self.truck_mod == "ew_kupol" else 1.0)
        if not fiber and self.ew_active_timer > 0.0 and dist_to_truck < ew_rad:
            jam_strength = 1.15 if self.truck_mod == "ew_kupol" else 1.0
            self.drone_jam_level = clamp((1.0 - (dist_to_truck / ew_rad) * 0.70) * jam_strength, 0.28, 1.0)
        elif not fiber and dist_to_truck < 30.0:
            self.drone_jam_level = clamp((30.0 - dist_to_truck) / 45.0, 0.0, 0.42)
        else:
            self.drone_jam_level = max(0.0, self.drone_jam_level - dt * 2.5)

        # Control jitter when jammed by EW
        ctrl_mult = 1.0 - 0.58 * self.drone_jam_level
        jam_yaw_glitch = math.sin(self.sim_time * 19.0) * 1.6 * self.drone_jam_level * dt if self.drone_jam_level > 0.30 else 0.0
        jam_roll_glitch = math.cos(self.sim_time * 15.0) * 1.8 * self.drone_jam_level if self.drone_jam_level > 0.30 else 0.0

        # Harder Quadcopter Rotational Inertia & Acro Attitude Hold
        acro = bool(inp.acro_mode)
        self.drone_acro = acro
        ang_resp = 21.0 if wind_gyro else 15.5

        # Yaw & Pitch rates driven by mouse/stick + cyclic W/S forward tilt
        safe_dt = max(dt, 1e-4)
        target_yaw_rate = (inp.yaw_delta / safe_dt) * ctrl_mult
        # W/S also nudges nose pitch down/up gently (+forward_Axis tilts nose down!)
        target_pitch_rate = (inp.pitch_delta / safe_dt) * ctrl_mult + inp.forward_Axis * 0.55 * ctrl_mult
        target_roll_rate = -inp.strafe_axis * (3.15 if acro else 2.65) * ctrl_mult + jam_roll_glitch

        self.drone_yaw_rate += (target_yaw_rate - self.drone_yaw_rate) * min(1.0, dt * ang_resp)
        self.drone_pitch_rate += (target_pitch_rate - self.drone_pitch_rate) * min(1.0, dt * ang_resp)
        self.drone_roll_rate += (target_roll_rate - self.drone_roll_rate) * min(1.0, dt * (12.0 if acro else 10.0))

        # Weather turbulence nudges drone attitude slightly (wind_gyro damps 75% of it)
        turb_scale = (0.25 if wind_gyro else 1.0) * (w.gust_strength / 6.5)
        gust_roll_torque = math.sin(self.sim_time * 3.1 + float(self.drone_pos[2]) * 0.05) * 0.38 * turb_scale * dt
        gust_pitch_torque = math.cos(self.sim_time * 2.7) * 0.24 * turb_scale * dt

        # Coordinated bank-to-yaw coupling when rolled
        bank_yaw_drift = -math.sin(self.drone_roll) * 0.48 * dt

        self.drone_yaw = wrap_angle(self.drone_yaw + self.drone_yaw_rate * dt + jam_yaw_glitch + bank_yaw_drift)
        self.drone_pitch = clamp(self.drone_pitch + self.drone_pitch_rate * dt + gust_pitch_torque, -1.35, 1.38)

        if acro:
            # Acro Mode: NO self-leveling! Drone holds whatever roll & pitch angle you set!
            self.drone_roll = clamp(self.drone_roll + self.drone_roll_rate * dt + gust_roll_torque, -1.48, 1.48)
        else:
            # Angle Assist: still uses real thrust vector & gravity, gently levels when sticks centered
            self.drone_roll = clamp(self.drone_roll + self.drone_roll_rate * dt + gust_roll_torque, -1.15, 1.15)
            if abs(inp.strafe_axis) < 0.06:
                self.drone_roll = move_toward(self.drone_roll, 0.0, dt * 2.2)
            if abs(inp.pitch_delta) < 1e-4 and abs(inp.forward_Axis) < 0.06:
                self.drone_pitch = move_toward(self.drone_pitch, 0.0, dt * 1.15)

        # True Tilted Rotor Thrust Vectoring (no magic auto-hover or sideways FPS strafe!)
        speed_mult = (
            (1.25 if sport else 1.0)
            * (0.88 if heavy_pg else 1.0)
            * (0.94 if fiber else 1.0)
            * (0.95 if lipo else 1.0)
        )
        boost_mult = (1.38 if sport else 1.20) if inp.boost else 1.0
        max_spd = DRONE_MAX_SPEED * speed_mult * boost_mult
        thrust_mag = DRONE_THRUST * speed_mult * boost_mult

        rot = euler_to_matrix(self.drone_yaw, self.drone_pitch, self.drone_roll)
        # Camera uptilt ~20 deg: when camera looks flat at horizon, frame + rotors tilt forward 0.36
        tilt_fwd = 0.36 + inp.forward_Axis * 0.42 + (0.12 if inp.boost else 0.0)
        local_thrust = np.array([0.0, 1.0, tilt_fwd], dtype=np.float32)
        local_thrust /= max(float(np.linalg.norm(local_thrust)), 1e-6)
        thrust_dir = rot @ local_thrust

        # Collective motor throttle: neutral is only ~64% of gravity -> pilot MUST manage throttle & pitch!
        hover_def = DRONE_GRAVITY * 0.64
        if inp.throttle >= 0.0:
            motor_accel = hover_def + inp.throttle * (thrust_mag - hover_def)
        else:
            motor_accel = max(DRONE_GRAVITY * 0.14, hover_def + inp.throttle * (hover_def * 0.82))
        if abs(inp.forward_Axis) > 0.05:
            # Slight cyclic throttle compensation when pressing W/S
            motor_accel += abs(inp.forward_Axis) * DRONE_GRAVITY * 0.30
        if inp.boost:
            motor_accel *= 1.16
        motor_accel = clamp(motor_accel, DRONE_GRAVITY * 0.12, thrust_mag)

        self.drone_throttle_pct = clamp((motor_accel / max(thrust_mag, 1.0)) * 100.0, 4.0, 100.0)

        accel = thrust_dir * motor_accel
        accel[1] -= DRONE_GRAVITY

        # Dynamic weather wind + gusts (wind_gyro reduces wind push by 75%!)
        wind_mult = 0.25 if wind_gyro else 1.0
        gust_x = math.sin(self.sim_time * 1.7 + float(self.drone_pos[2]) * 0.03) * w.gust_strength
        gust_y = math.cos(self.sim_time * 2.3) * (w.gust_strength * 0.28)
        gust_z = math.cos(self.sim_time * 1.3) * (w.gust_strength * 0.5)
        wind_vec = np.array(
            [w.wind_vec[0] + gust_x, w.wind_vec[1] + gust_y, w.wind_vec[2] + gust_z],
            dtype=np.float32,
        ) * wind_mult

        rel_vel = self.drone_vel - wind_vec * 0.62
        accel -= rel_vel * DRONE_DRAG

        self.drone_vel += accel * dt
        spd = float(np.linalg.norm(self.drone_vel))
        if spd > max_spd:
            self.drone_vel *= max_spd / spd

        self.drone_pos += self.drone_vel * dt
        self.drone_pos[0] = clamp(float(self.drone_pos[0]), -220.0, 220.0)
        self.drone_pos[1] = clamp(float(self.drone_pos[1]), -1.0, 165.0)
        self.drone_pos[2] = clamp(float(self.drone_pos[2]), -860.0, 860.0)

        # Battery consumption (higher throttle & boost drain battery faster)
        throttle_drain = 0.80 + 0.45 * (self.drone_throttle_pct / 100.0)
        boost_drain = (1.95 if sport else 1.65) if inp.boost else 1.0
        self.drone_battery -= w.drone_battery_drain_mult * throttle_drain * boost_drain * dt
        if self.drone_battery <= 0.0:
            self._destroy_drone_in_air("Села батарея FPV дрона!")
            return

        self.drone_detonate_cd = max(0.0, self.drone_detonate_cd - dt)

        # Manual remote warhead drop / cluster munition release
        if inp.detonate and self.drone_detonate_cd <= 0.0:
            if self.drone_mod == "cluster_munitions":
                self._release_cluster_bomblets()
            else:
                self._detonate_warhead("Дистанционный подрыв БЧ!", manual=True)
            return

        # Direct / Proximity hit on Fuel Truck hitbox
        truck_center = self.truck_pos + np.array([0.0, 2.15, -0.3], dtype=np.float32)
        rel = self.drone_pos - truck_center
        cy, sy = math.cos(-self.truck_yaw), math.sin(-self.truck_yaw)
        lx = rel[0] * cy + rel[2] * sy
        ly = rel[1]
        lz = -rel[0] * sy + rel[2] * cy
        if abs(lx) <= 2.15 and abs(ly) <= 2.25 and abs(lz) <= 6.1:
            self._detonate_warhead("ПРЯМОЕ ПОПАДАНИЕ FPV В БЕНЗОВОЗ!")
            return

        dx, dy, dz = float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])
        if dy <= 0.35:
            self._destroy_drone_in_air("FPV дрон упал в воды Керченского пролива!")
            return

        if abs(dx) <= ROAD_HALF_WIDTH + 0.6 and abs(dz) <= 800.0:
            if ROAD_DECK_Y - 1.8 <= dy <= ROAD_DECK_Y + 0.38:
                self._detonate_warhead("FPV разбился о полотно моста!")
                return

        for col in self.static_colliders:
            if (col.min_pt[0] <= dx <= col.max_pt[0] and
                    col.min_pt[1] <= dy <= col.max_pt[1] and
                    col.min_pt[2] <= dz <= col.max_pt[2]):
                self._detonate_warhead("FPV врезался в конструкцию моста!")
                return

    def _release_cluster_bomblets(self) -> None:
        self.drone_detonate_cd = 2.2
        offsets = [
            np.array([-1.2, -0.4, 0.0], dtype=np.float32),
            np.array([0.0, -0.5, 0.8], dtype=np.float32),
            np.array([1.2, -0.4, -0.4], dtype=np.float32),
        ]
        for off in offsets:
            bvel = self.drone_vel * 0.72 + np.array([off[0] * 3.2, -4.5, off[2] * 3.2], dtype=np.float32)
            self.projectiles.append(
                Projectile(
                    pos=self.drone_pos.copy() + np.array([0.0, -0.5, 0.0], dtype=np.float32),
                    prev_pos=self.drone_pos.copy(),
                    vel=bvel.astype(np.float32),
                    damage=36.0,
                    hit_radius=1.0,
                    drag=0.12,
                    gravity_mult=1.0,
                    ttl=3.5,
                    initial_speed=float(np.linalg.norm(bvel)),
                    color=(255, 110, 45),
                    weapon_name="Кассетный заряд",
                    kind="bomblet",
                )
            )
        self.last_event_ru = "Сброшена кассета из 3 суббоеприпасов!"

    def _destroy_drone_in_air(self, reason_ru: str) -> None:
        if not self.drone_alive:
            return
        self.explosions.append(
            ExplosionFX(
                pos=(float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])),
                radius=4.5,
                ttl=0.65,
                max_ttl=0.65,
            )
        )
        self.drone_alive = False
        self.drone_lives_left -= 1
        self.drone_respawn_timer = DRONE_RESPAWN_DELAY
        self.last_event_ru = reason_ru

    def _apply_warhead_damage_to_truck(self, origin: np.ndarray, blast_radius: float, base_dmg: float, reason_ru: str) -> str:
        """Applies warhead blast damage at `origin` considering cage, tandem, ERA & auto-extinguisher."""
        cage = self.truck_mod == "anti_drone_cage"
        tandem = self.drone_mod == "tandem_warhead"
        auto_ext = self.truck_mod == "auto_extinguisher"

        if cage and not tandem:
            base_dmg *= 0.65

        truck_center = self.truck_pos + np.array([0.0, 2.15, -0.4], dtype=np.float32)
        rel = origin - truck_center
        cy, sy = math.cos(-self.truck_yaw), math.sin(-self.truck_yaw)
        lx = clamp(float(rel[0] * cy + rel[2] * sy), -1.6, 1.6)
        ly = clamp(float(rel[1]), -1.8, 1.8)
        lz = clamp(float(-rel[0] * sy + rel[2] * cy), -5.5, 5.2)
        closest_local = np.array([lx, ly, lz], dtype=np.float32)
        cy_f, sy_f = math.cos(self.truck_yaw), math.sin(self.truck_yaw)
        closest_world = truck_center + np.array(
            [closest_local[0] * cy_f + closest_local[2] * sy_f,
             closest_local[1],
             -closest_local[0] * sy_f + closest_local[2] * cy_f],
            dtype=np.float32,
        )
        dist = float(np.linalg.norm(origin - closest_world))

        if dist <= blast_radius:
            falloff = clamp(1.0 - (dist / blast_radius) * 0.78, 0.22, 1.0)
            dealt = base_dmg * falloff
            if self.truck_reactive_ready and dealt > 5.0:
                self.truck_reactive_ready = False
                dealt *= 0.65 if tandem else 0.35
                reason_ru = f"ДЗ «Контакт» отбила удар! {reason_ru}"
            self.truck_hp = max(0.0, self.truck_hp - dealt)
            self.truck_last_hit_timer = 0.0
            self.truck_speed *= 0.93 if auto_ext else 0.85
            reason_ru = f"{reason_ru} (-{int(round(dealt))} HP бензовозу!)"
        else:
            reason_ru = f"{reason_ru} (заряд не достал до цистерны!)"
        return reason_ru

    def _explode_point(self, origin: np.ndarray, radius: float, base_dmg: float, reason_ru: str) -> None:
        msg = self._apply_warhead_damage_to_truck(origin, radius, base_dmg, reason_ru)
        self.explosions.append(
            ExplosionFX(
                pos=(float(origin[0]), float(origin[1]), float(origin[2])),
                radius=radius,
                ttl=0.85,
                max_ttl=0.85,
            )
        )
        self.last_event_ru = msg

    def _detonate_warhead(self, reason_ru: str, manual: bool = False) -> None:
        if not self.drone_alive:
            return
        frag = self.drone_mod == "airburst_frag"
        shaped = self.drone_mod == "shaped_charge"
        tandem = self.drone_mod == "tandem_warhead"

        blast_radius = DRONE_EXPLOSION_RADIUS * (1.55 if frag else 1.0)
        base_dmg = DRONE_DIRECT_DAMAGE * (1.40 if shaped else (1.20 if tandem else 1.0))
        if manual:
            base_dmg *= 0.85

        reason_ru = self._apply_warhead_damage_to_truck(self.drone_pos, blast_radius, base_dmg, reason_ru)

        self.explosions.append(
            ExplosionFX(
                pos=(float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])),
                radius=blast_radius,
                ttl=1.05,
                max_ttl=1.05,
            )
        )

        if manual:
            self.drone_detonate_cd = DRONE_DETONATE_COOLDOWN
        else:
            self.drone_alive = False
            self.drone_lives_left -= 1
            self.drone_respawn_timer = DRONE_RESPAWN_DELAY
        self.last_event_ru = reason_ru

    # -----------------------------------------------------------------------
    # Round Victory / Defeat Evaluation
    # -----------------------------------------------------------------------
    def _check_round_conditions(self) -> None:
        if self.truck_hp <= 0.0:
            self.explosions.append(
                ExplosionFX(
                    pos=(float(self.truck_pos[0]), float(self.truck_pos[1] + 2.5), float(self.truck_pos[2])),
                    radius=24.0,
                    ttl=2.2,
                    max_ttl=2.2,
                )
            )
            self._finish_round("drone", "Цистерна бензовоза уничтожена взрывом FPV!")
            return

        if float(self.truck_pos[2]) >= BRIDGE_FINISH_Z:
            self._finish_round("truck", "Бензовоз прорвался через Крымский мост до финиша!")
            return

        if not self.drone_alive and self.drone_lives_left <= 0:
            self._finish_round("truck", "Все FPV дроны сбиты или потеряны! Бензовоз уцелел!")
            return

        if self.phase_timer <= 0.0:
            self._finish_round("truck", "Время вышло — оператор FPV не успел поразить цель!")

    def _finish_round(self, winner: str, reason_ru: str) -> None:
        self.phase = "ROUND_END"
        self.phase_timer = ROUND_END_DELAY
        self.round_winner = winner
        self.round_reason_ru = reason_ru
        self.last_event_ru = reason_ru
        if winner == "drone":
            self.drone_score += 1
        else:
            self.truck_score += 1

    # -----------------------------------------------------------------------
    # Network State Serialization (Compact JSON-friendly dict)
    # -----------------------------------------------------------------------
    def get_snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "round_number": self.round_number,
            "drone_score": self.drone_score,
            "truck_score": self.truck_score,
            "phase_timer": round(self.phase_timer, 2),
            "round_winner": self.round_winner,
            "round_reason_ru": self.round_reason_ru,
            "match_winner": self.match_winner,
            "weather_id": self.weather_id,
            "lightning_flash": round(self.lightning_flash, 2),
            "sim_time": round(self.sim_time, 2),
            "drone_mod": self.drone_mod,
            "truck_mod": self.truck_mod,
            "drone_draft_ready": self.drone_draft_ready,
            "truck_draft_ready": self.truck_draft_ready,
            "truck": {
                "pos": [round(float(x), 3) for x in self.truck_pos],
                "vel": [round(float(x), 2) for x in self.truck_vel],
                "speed": round(self.truck_speed, 2),
                "yaw": round(self.truck_yaw, 4),
                "hp": round(self.truck_hp, 1),
                "max_hp": round(self.truck_max_hp, 1),
                "reactive_ready": self.truck_reactive_ready,
                "turret_yaw": round(self.turret_yaw, 4),
                "turret_pitch": round(self.turret_pitch, 4),
                "turret_heat": round(self.turret_heat, 3),
                "turret_lock": round(self.turret_lock, 3),
                "turret_overheated": self.turret_overheated,
                "shotgun_cd": round(self.shotgun_cd, 2),
                "ew_active_timer": round(self.ew_active_timer, 2),
                "ew_cd": round(self.ew_cd, 2),
                "smoke_cd": round(self.smoke_cd, 2),
            },
            "drone": {
                "pos": [round(float(x), 3) for x in self.drone_pos],
                "vel": [round(float(x), 2) for x in self.drone_vel],
                "yaw": round(self.drone_yaw, 4),
                "pitch": round(self.drone_pitch, 4),
                "roll": round(self.drone_roll, 4),
                "throttle_pct": round(self.drone_throttle_pct, 1),
                "hp": round(self.drone_hp, 1),
                "max_hp": round(self.drone_max_hp, 1),
                "battery": round(self.drone_battery, 1),
                "max_battery": round(self.drone_max_battery, 1),
                "lives_left": self.drone_lives_left,
                "alive": self.drone_alive,
                "respawn_timer": round(self.drone_respawn_timer, 2),
                "jam_level": round(self.drone_jam_level, 3),
                "detonate_cd": round(self.drone_detonate_cd, 2),
                "acro": bool(self.drone_acro),
            },
            "projectiles": [
                {
                    "s": [round(float(c), 2) for c in p.prev_pos],
                    "e": [round(float(c), 2) for c in p.pos],
                    "c": list(p.color),
                    "k": p.kind,
                }
                for p in self.projectiles[-36:]
            ],
            "tracers": [
                {"s": [round(c, 2) for c in t.start], "e": [round(c, 2) for c in t.end], "c": list(t.color), "ttl": round(t.ttl, 2)}
                for t in self.tracers[-18:]
            ],
            "explosions": [
                {"p": [round(c, 2) for c in e.pos], "r": round(e.radius, 2), "ttl": round(e.ttl, 2), "m": round(e.max_ttl, 2)}
                for e in self.explosions
            ],
            "smokes": [
                {"p": [round(c, 2) for c in s.pos], "r": round(s.radius, 2), "ttl": round(s.ttl, 2)}
                for s in self.smokes
            ],
            "last_event_ru": self.last_event_ru,
        }

    def apply_snapshot(self, snap: dict) -> None:
        self.phase = snap["phase"]
        self.round_number = snap["round_number"]
        self.drone_score = snap["drone_score"]
        self.truck_score = snap["truck_score"]
        self.phase_timer = snap["phase_timer"]
        self.round_winner = snap["round_winner"]
        self.round_reason_ru = snap["round_reason_ru"]
        self.match_winner = snap["match_winner"]
        self.weather_id = snap["weather_id"]
        self.lightning_flash = snap["lightning_flash"]
        self.sim_time = snap["sim_time"]
        self.drone_mod = snap["drone_mod"]
        self.truck_mod = snap["truck_mod"]
        self.drone_draft_ready = snap["drone_draft_ready"]
        self.truck_draft_ready = snap["truck_draft_ready"]

        t = snap["truck"]
        self.truck_pos[:] = t["pos"]
        self.truck_vel[:] = t["vel"]
        self.truck_speed = t["speed"]
        self.truck_yaw = t["yaw"]
        self.truck_hp = t["hp"]
        self.truck_max_hp = t["max_hp"]
        self.truck_reactive_ready = bool(t.get("reactive_ready", False))
        self.turret_yaw = t["turret_yaw"]
        self.turret_pitch = t["turret_pitch"]
        self.turret_heat = t["turret_heat"]
        self.turret_lock = t.get("turret_lock", 0.0)
        self.turret_overheated = t["turret_overheated"]
        self.shotgun_cd = t["shotgun_cd"]
        self.ew_active_timer = t["ew_active_timer"]
        self.ew_cd = t["ew_cd"]
        self.smoke_cd = t["smoke_cd"]

        d = snap["drone"]
        self.drone_pos[:] = d["pos"]
        self.drone_vel[:] = d["vel"]
        self.drone_yaw = d["yaw"]
        self.drone_pitch = d["pitch"]
        self.drone_roll = d["roll"]
        self.drone_throttle_pct = float(d.get("throttle_pct", 35.0))
        self.drone_hp = d["hp"]
        self.drone_max_hp = d["max_hp"]
        self.drone_battery = d["battery"]
        self.drone_max_battery = d["max_battery"]
        self.drone_lives_left = d["lives_left"]
        self.drone_alive = d["alive"]
        self.drone_respawn_timer = d["respawn_timer"]
        self.drone_jam_level = d["jam_level"]
        self.drone_detonate_cd = d.get("detonate_cd", 0.0)
        self.drone_acro = bool(d.get("acro", True))

        self.net_proj_lines = [
            TracerFX(start=tuple(x["s"]), end=tuple(x["e"]), color=tuple(x["c"]), ttl=0.05, kind=x.get("k", "turret"))
            for x in snap.get("projectiles", [])
        ]
        self.tracers = [
            TracerFX(start=tuple(x["s"]), end=tuple(x["e"]), color=tuple(x["c"]), ttl=x["ttl"])
            for x in snap.get("tracers", [])
        ]
        self.explosions = [
            ExplosionFX(pos=tuple(x["p"]), radius=x["r"], ttl=x["ttl"], max_ttl=x["m"])
            for x in snap.get("explosions", [])
        ]
        self.smokes = [
            SmokeCloud(pos=tuple(x["p"]), radius=x["r"], ttl=x["ttl"])
            for x in snap.get("smokes", [])
        ]
        self.last_event_ru = snap.get("last_event_ru", "")
