"""
Authoritative 3D Physics & Match Simulation for Crimean Bridge 1v1:
- Match state machine: LOBBY -> MODIFIER_DRAFT -> ACTIVE_ROUND -> ROUND_END -> MATCH_OVER
- Dynamic weather rotation every round (guaranteed distinct consecutive presets)
- Modifiers applied to both FPV Drone and Fuel Truck
- Anti-FPV weapons: Shotgun spread raycast, AA Turret tracers, EW Jammer (РЭБ), Smoke screen
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    BRIDGE_FINISH_Z,
    BRIDGE_START_Z,
    DRONE_BASE_HP,
    DRONE_BASE_LIVES,
    DRONE_BATTERY_SECONDS,
    DRONE_DIRECT_DAMAGE,
    DRONE_DETONATE_COOLDOWN,
    DRONE_DRAG,
    DRONE_EXPLOSION_RADIUS,
    DRONE_GRAVITY,
    DRONE_MAX_SPEED,
    DRONE_MODIFIERS,
    DRONE_RESPAWN_DELAY,
    DRONE_THRUST,
    DRONE_TRIGGER_DIST,
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
    ray_sphere_intersect,
    wrap_angle,
)


@dataclass
class DroneInput:
    throttle: float = 0.0        # -1..+1 (collective / climb / forward thrust)
    forward_Axis: float = 0.0    # -1..+1 pitch/surge forward-back
    strafe_axis: float = 0.0     # -1..+1 roll/strafe left-right
    yaw_delta: float = 0.0       # radians turn delta this tick (mouse/stick)
    pitch_delta: float = 0.0     # radians look pitch delta this tick
    boost: bool = False          # sprint / nitro motors
    detonate: bool = False       # manual airburst trigger
    acro_mode: bool = False      # toggle acro vs stabilized horizon
    pick_modifier: str = ""      # modifier id selected during draft
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
    pick_modifier: str = ""      # modifier id selected during draft
    ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TruckInput":
        valid = {k: v for k, v in d.items() if k in TruckInput.__dataclass_fields__}
        return TruckInput(**valid)


@dataclass
class TracerFX:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    color: Tuple[int, int, int]
    ttl: float


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

        # Weather schedule (shuffled bag so every round has a different weather!)
        self._weather_bag: List[str] = []
        self.weather_id: str = "clear_day"
        self.lightning_flash: float = 0.0
        self.sim_time: float = 0.0

        # Active Modifiers per player
        self.drone_mod: str = DRONE_MODIFIERS[0].id
        self.truck_mod: str = TRUCK_MODIFIERS[0].id
        self.drone_draft_ready: bool = False
        self.truck_draft_ready: bool = False

        # Fuel Truck State
        self.truck_pos = np.array([3.2, ROAD_DECK_Y, BRIDGE_START_Z], dtype=np.float32)
        self.truck_vel = np.zeros(3, dtype=np.float32)
        self.truck_speed: float = 0.0
        self.truck_yaw: float = 0.0          # 0 = heading +Z along bridge
        self.truck_hp: float = TRUCK_BASE_HP
        self.truck_max_hp: float = TRUCK_BASE_HP
        self.turret_yaw: float = math.pi     # Initially looking backward (-Z) toward approaching drone
        self.turret_pitch: float = -0.18     # Slightly elevated into sky
        self.turret_heat: float = 0.0
        self.turret_lock: float = 0.0        # 0..1 tracking lock on the FPV</new_string>
        self.turret_overheated: bool = False
        self.turret_cd: float = 0.0
        self.shotgun_cd: float = 0.0
        self.ew_active_timer: float = 0.0
        self.ew_cd: float = 0.0
        self.smoke_cd: float = 0.0

        # FPV Drone State
        self.drone_pos = np.array([0.0, ROAD_DECK_Y + 26.0, BRIDGE_START_Z + 420.0], dtype=np.float32)
        self.drone_vel = np.zeros(3, dtype=np.float32)
        self.drone_yaw: float = math.pi      # Facing -Z toward oncoming truck
        self.drone_pitch: float = 0.08
        self.drone_roll: float = 0.0
        self.drone_hp: float = DRONE_BASE_HP
        self.drone_max_hp: float = DRONE_BASE_HP
        self.drone_battery: float = DRONE_BATTERY_SECONDS
        self.drone_max_battery: float = DRONE_BATTERY_SECONDS
        self.drone_lives_left: int = DRONE_BASE_LIVES
        self.drone_alive: bool = True
        self.drone_respawn_timer: float = 0.0
        self.drone_jam_level: float = 0.0    # 0..1 EW interference strength

        # Transient FX synced to clients
        self.tracers: List[TracerFX] = []
        self.explosions: List[ExplosionFX] = []
        self.smokes: List[SmokeCloud] = []
        self.last_event_ru: str = "Раунд 1: выберите модификаторы перед боем!"

        self._pick_next_weather()
        self._init_round_entities()

    def _build_colliders(self) -> List[AABB]:
        cols = list(self.barriers)
        # Main shipping arch overhead crossbeams & ribs approximation
        arch_steps = 18
        for s in range(arch_steps):
            u0 = -1.0 + 2.0 * (s / arch_steps)
            u1 = -1.0 + 2.0 * ((s + 1) / arch_steps)
            z0 = u0 * 115.0
            z1 = u1 * 115.0
            y0 = ROAD_DECK_Y + (63.0 - ROAD_DECK_Y) * (1.0 - u0 * u0)
            y1 = ROAD_DECK_Y + (63.0 - ROAD_DECK_Y) * (1.0 - u1 * u1)
            for side_x in (-ROAD_HALF_WIDTH - 0.6, ROAD_HALF_WIDTH + 0.6):
                cols.append(
                    AABB(
                        min_pt=np.array([side_x - 1.3, min(y0, y1) - 1.3, min(z0, z1) - 1.3], dtype=np.float32),
                        max_pt=np.array([side_x + 1.3, max(y0, y1) + 1.3, max(z0, z1) + 1.3], dtype=np.float32),
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
            # Avoid repeating the exact same weather across bag boundary
            if ids[0] == self.weather_id and len(ids) > 1:
                ids[0], ids[-1] = ids[-1], ids[0]
            self._weather_bag = ids
        self.weather_id = self._weather_bag.pop(0)

    def reset_full_match(self) -> None:
        self.round_number = 1
        self.drone_score = 0
        self.truck_score = 0
        self.match_winner = ""
        self._weather_bag.clear()
        self.start_modifier_draft(advance_weather=True)

    def start_modifier_draft(self, advance_weather: bool = True) -> None:
        if advance_weather:
            self._pick_next_weather()
        self.phase = "MODIFIER_DRAFT"
        self.phase_timer = MODIFIER_PICK_TIME
        self.drone_draft_ready = False
        self.truck_draft_ready = False
        self.round_winner = ""
        self.round_reason_ru = ""
        self._init_round_entities()
        self.last_event_ru = f"Раунд {self.round_number} | Погода: {self.weather.name_ru}"

    def _init_round_entities(self) -> None:
        self.tracers.clear()
        self.explosions.clear()
        self.smokes.clear()

        # Apply Truck Modifiers
        self.truck_max_hp = TRUCK_BASE_HP
        self.truck_hp = self.truck_max_hp
        self.truck_pos = np.array([3.2, ROAD_DECK_Y, BRIDGE_START_Z], dtype=np.float32)
        self.truck_vel = np.zeros(3, dtype=np.float32)
        self.truck_speed = 12.0  # Rolling start onto the bridge
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
        hp_mult = 1.45 if self.drone_mod == "kevlar_frame" else 1.0
        bat_mult = 1.20 if self.drone_mod == "fiber_optic" else 1.0

        self.drone_max_hp = DRONE_BASE_HP * hp_mult
        self.drone_hp = self.drone_max_hp
        self.drone_max_battery = DRONE_BATTERY_SECONDS * bat_mult
        self.drone_battery = self.drone_max_battery
        self.drone_alive = True
        self.drone_respawn_timer = 0.0
        self.drone_jam_level = 0.0
        self.drone_detonate_cd = 0.0

        # Spawn just beyond the iconic white Crimean Bridge arches facing the oncoming truck!
        spawn_z = clamp(float(self.truck_pos[2]) + 330.0, 95.0, BRIDGE_FINISH_Z + 40.0)
        spawn_x = float(self.rng.choice([-8.5, 0.0, 8.5, 19.0]))
        spawn_y = ROAD_DECK_Y + (28.0 if initial else 34.0)
        self.drone_pos = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float32)
        self.drone_vel = np.array([0.0, 0.0, -14.0], dtype=np.float32)
        # Aim toward truck
        dx = float(self.truck_pos[0] - self.drone_pos[0])
        dz = float(self.truck_pos[2] - self.drone_pos[2])
        self.drone_yaw = math.atan2(dx, dz)
        self.drone_pitch = 0.10
        self.drone_roll = 0.0
        self.drone_acro = False

    def begin_active_round(self) -> None:
        self.phase = "ACTIVE_ROUND"
        self.phase_timer = ROUND_TIME_LIMIT
        self._init_round_entities()
        self.last_event_ru = f"БОЙ НАЧАЛСЯ! Погода: {self.weather.name_ru}"

    # -----------------------------------------------------------------------
    # Main Simulation Step
    # -----------------------------------------------------------------------
    def step(self, dt: float, drone_in: DroneInput, truck_in: TruckInput) -> None:
        self.sim_time += dt
        self._tick_fx(dt)

        # Weather lightning ambience
        if self.lightning_flash > 0.0:
            self.lightning_flash = max(0.0, self.lightning_flash - dt * 3.2)
        elif self.weather.lightning_chance > 0.0:
            if self.rng.random() < self.weather.lightning_chance * dt:
                self.lightning_flash = 1.0

        if self.phase == "MODIFIER_DRAFT":
            if drone_in.pick_modifier in {m.id for m in DRONE_MODIFIERS}:
                self.drone_mod = drone_in.pick_modifier
            if truck_in.pick_modifier in {m.id for m in TRUCK_MODIFIERS}:
                self.truck_mod = truck_in.pick_modifier
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
        self._step_truck(dt, truck_in)
        self._step_drone(dt, drone_in)
        self._check_round_conditions()

    def _tick_fx(self, dt: float) -> None:
        self.tracers = [t for t in self.tracers if (setattr(t, "ttl", t.ttl - dt) or t.ttl > 0.0)]
        self.explosions = [e for e in self.explosions if (setattr(e, "ttl", e.ttl - dt) or e.ttl > 0.0)]
        self.smokes = [s for s in self.smokes if (setattr(s, "ttl", s.ttl - dt) or s.ttl > 0.0)]

    # -----------------------------------------------------------------------
    # Truck Driving & Anti-FPV Weapons
    # -----------------------------------------------------------------------
    def _step_truck(self, dt: float, inp: TruckInput) -> None:
        w = self.weather
        v12 = self.truck_mod == "v12_biturbo"
        max_spd = TRUCK_BASE_MAX_SPEED * (1.26 if v12 else 1.0)
        accel_rate = TRUCK_BASE_ACCEL * (1.35 if v12 else 1.0)
        grip = min(1.0, w.truck_grip * (1.35 if v12 else 1.0))

        # Longitudinal acceleration / braking
        if inp.accel >= 0.0:
            self.truck_speed += inp.accel * accel_rate * grip * dt
        else:
            if self.truck_speed > 0.5:
                self.truck_speed += inp.accel * TRUCK_BASE_BRAKE * grip * dt
            else:
                self.truck_speed += inp.accel * (accel_rate * 0.55) * grip * dt

        # Natural rolling drag
        self.truck_speed -= self.truck_speed * 0.14 * dt
        self.truck_speed = clamp(self.truck_speed, -9.0, max_spd)

        # Steering (slippery road adds slight drift / sluggish turn response)
        speed_factor = clamp(abs(self.truck_speed) / 8.0, 0.25, 1.0)
        reverse_sign = -1.0 if self.truck_speed < -0.2 else 1.0
        max_yaw_from_bridge = 0.62  # Prevent full U-turn spinout
        self.truck_yaw += inp.steer * TRUCK_STEER_RATE * grip * speed_factor * reverse_sign * dt
        self.truck_yaw = clamp(self.truck_yaw, -max_yaw_from_bridge, max_yaw_from_bridge)

        # Move along bridge deck
        vx = math.sin(self.truck_yaw) * self.truck_speed
        vz = math.cos(self.truck_yaw) * self.truck_speed
        self.truck_vel[:] = (vx, 0.0, vz)
        self.truck_pos[0] += vx * dt
        self.truck_pos[2] += vz * dt
        self.truck_pos[1] = ROAD_DECK_Y

        # Guardrail clamping & scrape slowdown
        guard_limit = ROAD_HALF_WIDTH - 2.1
        if abs(float(self.truck_pos[0])) > guard_limit:
            self.truck_pos[0] = clamp(float(self.truck_pos[0]), -guard_limit, guard_limit)
            self.truck_speed *= 0.88
            self.truck_yaw *= 0.7

        self.truck_pos[2] = max(BRIDGE_START_Z - 20.0, float(self.truck_pos[2]))

        # Concrete barrier collision check (push aside + speed loss, minor bump)
        tx, tz = float(self.truck_pos[0]), float(self.truck_pos[2])
        half_tw, half_tl = 1.55, 5.6
        for b in self.barriers:
            if (tx + half_tw > b.min_pt[0] and tx - half_tw < b.max_pt[0] and
                    tz + half_tl > b.min_pt[2] and tz - half_tl < b.max_pt[2]):
                # Push laterally away from barrier center
                bcx = 0.5 * float(b.min_pt[0] + b.max_pt[0])
                push_dir = 1.0 if tx >= bcx else -1.0
                self.truck_pos[0] = bcx + push_dir * (0.5 * float(b.max_pt[0] - b.min_pt[0]) + half_tw + 0.15)
                self.truck_speed *= 0.55
                self.last_event_ru = "Бензовоз задел бетонный блок!"

        # Gunner / Turret Aiming
        self.turret_yaw = wrap_angle(self.turret_yaw + inp.aim_yaw_delta)
        # Pitch: negative = looking up into sky (forward_from_angles uses -sin(pitch) for Y)
        self.turret_pitch = clamp(self.turret_pitch + inp.aim_pitch_delta, -1.35, 0.28)

        # Cooldowns
        self.turret_cd = max(0.0, self.turret_cd - dt)
        self.shotgun_cd = max(0.0, self.shotgun_cd - dt)
        self.ew_cd = max(0.0, self.ew_cd - dt)
        self.smoke_cd = max(0.0, self.smoke_cd - dt)
        if self.ew_active_timer > 0.0:
            self.ew_active_timer = max(0.0, self.ew_active_timer - dt)

        # Turret heat cooling
        cool_mult = 1.4 if self.truck_mod == "twin_zu_turret" else 1.0
        self.turret_heat = max(0.0, self.turret_heat - TURRET_COOL_RATE * cool_mult * dt)
        if self.turret_overheated and self.turret_heat < 0.25:
            self.turret_overheated = False

        muzzle_pos = self.get_turret_muzzle_pos()
        aim_dir = forward_from_angles(self.turret_yaw, self.turret_pitch)

        # Tracking lock: the gunner needs to keep the FPV inside the reticle cone for a while
        # before the rounds start to bite. This gives the drone a chance to break the lock.
        drone_range = 9999.0
        if self.drone_alive:
            to_drone = self.drone_pos - muzzle_pos
            drone_range = float(np.linalg.norm(to_drone))
            if drone_range > 1e-3:
                ang_err = math.acos(clamp(float(np.dot(aim_dir, to_drone / drone_range)), -1.0, 1.0))
                if ang_err < TURRET_LOCK_CONE and drone_range <= TURRET_RANGE:
                    self.turret_lock = min(1.0, self.turret_lock + dt / TURRET_LOCK_TIME)
                else:
                    self.turret_lock = max(0.0, self.turret_lock - dt * 1.6)
        else:
            self.turret_lock = max(0.0, self.turret_lock - dt * 1.6)
        lock_mult = 0.22 + 0.78 * self.turret_lock

        # 1) Rapid Roof AA Turret
        if inp.fire_turret and not self.turret_overheated and self.turret_cd <= 0.0:
            twin = self.truck_mod == "twin_zu_turret"
            self.turret_cd = TURRET_FIRE_INTERVAL * (0.58 if twin else 1.0)
            self.turret_heat += TURRET_HEAT_PER_SHOT * (0.60 if twin else 1.0)
            if self.turret_heat >= 1.0:
                self.turret_heat = 1.0
                self.turret_overheated = True
                self.last_event_ru = "Перегрев зенитной турели!"

            spread_rad = math.radians(1.35 if twin else 1.65)
            shot_dir = self._jitter_dir(aim_dir, spread_rad)
            # A small FPV is harder to hit at long range: the proximity hitbox shrinks with distance
            hit_radius = 1.05 + 0.35 * clamp(1.0 - drone_range / max(TURRET_RANGE, 1.0), 0.0, 1.0)
            self._fire_anti_drone_ray(
                origin=muzzle_pos,
                direction=shot_dir,
                max_range=TURRET_RANGE,
                damage=TURRET_DMG * (1.22 if twin else 1.0) * lock_mult,
                hit_radius=hit_radius,
                tracer_color=(255, 215, 75),
                tracer_ttl=0.11,
                weapon_name="ЗПУ турель",
            )

        # 2) Anti-FPV Buckshot Shotgun ("Перехват")
        if inp.fire_shotgun and self.shotgun_cd <= 0.0:
            buck_mod = self.truck_mod == "buckshot_radar"
            self.shotgun_cd = SHOTGUN_COOLDOWN * (0.82 if buck_mod else 1.0)
            pellets = SHOTGUN_PELLETS * (2 if buck_mod else 1)
            spread_rad = math.radians(SHOTGUN_SPREAD_DEG * (1.18 if buck_mod else 1.0))
            for _ in range(pellets):
                pdir = self._jitter_dir(aim_dir, spread_rad)
                self._fire_anti_drone_ray(
                    origin=muzzle_pos,
                    direction=pdir,
                    max_range=SHOTGUN_RANGE * (1.15 if buck_mod else 1.0),
                    damage=SHOTGUN_PELLET_DMG,
                    hit_radius=1.35,
                    tracer_color=(255, 155, 60),
                    tracer_ttl=0.16,
                    weapon_name="Дробовик 12к",
                )

        # 3) EW Jammer ("РЭБ Купол")
        if inp.trigger_ew and self.ew_cd <= 0.0:
            kupol = self.truck_mod == "ew_kupol"
            self.ew_active_timer = EW_JAMMER_DURATION * (1.35 if kupol else 1.0)
            self.ew_cd = EW_JAMMER_COOLDOWN * (0.72 if kupol else 1.0)
            self.last_event_ru = "Водитель включил купол РЭБ!"

        # 4) Smoke Screen ("Аэрозольная завеса")
        if inp.trigger_smoke and self.smoke_cd <= 0.0:
            buck_mod = self.truck_mod == "buckshot_radar"
            self.smoke_cd = SMOKE_COOLDOWN * (0.55 if buck_mod else 1.0)
            rear_pos = self.truck_pos - np.array([math.sin(self.truck_yaw) * 4.0, -2.2, math.cos(self.truck_yaw) * 4.0], dtype=np.float32)
            self.smokes.append(
                SmokeCloud(
                    pos=(float(rear_pos[0]), float(rear_pos[1]), float(rear_pos[2])),
                    radius=SMOKE_RADIUS,
                    ttl=SMOKE_DURATION,
                )
            )
            self.last_event_ru = "Поставлена дымовая завеса!"

    def get_turret_muzzle_pos(self) -> np.ndarray:
        # Roof of the truck cab (+3.3m ahead of truck center, +3.65m up)
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

    def _fire_anti_drone_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_range: float,
        damage: float,
        hit_radius: float,
        tracer_color: Tuple[int, int, int],
        tracer_ttl: float,
        weapon_name: str,
    ) -> None:
        end_pt = origin + direction * max_range
        if self.drone_alive:
            hit_t = ray_sphere_intersect(origin, direction, self.drone_pos, hit_radius, max_range)
            if hit_t is not None:
                end_pt = origin + direction * hit_t
                self.drone_hp -= damage
                # Knockback impulse on drone!
                self.drone_vel += direction * (damage * 0.35)
                if self.drone_hp <= 0.0:
                    self._destroy_drone_in_air(f"СБИТ FPV ДРОН ({weapon_name})!")

        self.tracers.append(
            TracerFX(
                start=(float(origin[0]), float(origin[1]), float(origin[2])),
                end=(float(end_pt[0]), float(end_pt[1]), float(end_pt[2])),
                color=tracer_color,
                ttl=tracer_ttl,
            )
        )

    # -----------------------------------------------------------------------
    # FPV Drone Flight, Wind, EW Jamming & Detonation
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

        # EW Jammer calculation
        dist_to_truck = float(np.linalg.norm(self.drone_pos - (self.truck_pos + np.array([0.0, 2.2, 0.0], dtype=np.float32))))
        ew_rad = EW_JAMMER_RADIUS * (1.60 if self.truck_mod == "ew_kupol" else 1.0)
        if not fiber and self.ew_active_timer > 0.0 and dist_to_truck < ew_rad:
            self.drone_jam_level = clamp(1.0 - (dist_to_truck / ew_rad) * 0.75, 0.25, 1.0)
        elif not fiber and dist_to_truck < 28.0:
            # Passive close-range analog video snow near roof dome
            self.drone_jam_level = clamp((28.0 - dist_to_truck) / 45.0, 0.0, 0.38)
        else:
            self.drone_jam_level = max(0.0, self.drone_jam_level - dt * 2.5)

        # Control jitter when jammed by EW
        ctrl_mult = 1.0 - 0.55 * self.drone_jam_level
        jam_yaw_glitch = math.sin(self.sim_time * 19.0) * 1.4 * self.drone_jam_level * dt if self.drone_jam_level > 0.35 else 0.0
        jam_roll_glitch = math.cos(self.sim_time * 15.0) * 0.45 * self.drone_jam_level

        self.drone_yaw = wrap_angle(self.drone_yaw + inp.yaw_delta * ctrl_mult + jam_yaw_glitch)
        self.drone_pitch = clamp(self.drone_pitch + inp.pitch_delta * ctrl_mult, -1.25, 1.35)

        # Flight mode: "Acro" (manual attitude, no self-levelling, more agile) vs "Stabilized"
        acro = bool(inp.acro_mode)
        self.drone_acro = acro
        if acro:
            target_roll = clamp(-inp.strafe_axis * 0.95 + jam_roll_glitch, -1.20, 1.20)
            self.drone_roll += (target_roll - self.drone_roll) * min(1.0, dt * 12.0)
        else:
            target_roll = clamp(-inp.strafe_axis * 0.68 + jam_roll_glitch, -0.95, 0.95)
            self.drone_roll += (target_roll - self.drone_roll) * min(1.0, dt * 9.5)
            # Slow self-levelling of the nose when the pilot does not hold pitch
            if abs(inp.pitch_delta) < 1e-4 and abs(inp.forward_Axis) < 0.05:
                self.drone_pitch = move_toward(self.drone_pitch, 0.0, dt * 0.85)

        # Compute 3D thrust vector
        speed_mult = (1.28 if sport else 1.0) * (0.88 if heavy_pg else 1.0) * (1.08 if acro else 1.0)
        boost_mult = (1.34 if sport else 1.18) if inp.boost else 1.0
        max_spd = DRONE_MAX_SPEED * speed_mult * boost_mult
        thrust_mag = DRONE_THRUST * speed_mult * boost_mult

        fwd = forward_from_angles(self.drone_yaw, self.drone_pitch)
        right = np.array([math.cos(self.drone_yaw), 0.0, -math.sin(self.drone_yaw)], dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Hybrid intuitive FPV flight model:
        # forward_Axis drives along camera look vector (dive / climb where aiming!)
        # throttle adds collective lift (hover compensation + vertical climb/sink)
        # strafe_axis slides sideways for lining up moving truck
        accel = np.zeros(3, dtype=np.float32)
        accel += world_up * (DRONE_GRAVITY * 0.96 + inp.throttle * thrust_mag * 0.72)
        accel += fwd * (inp.forward_Axis * thrust_mag * 1.08 + (0.35 * thrust_mag if inp.boost else 0.0))
        accel += right * (inp.strafe_axis * thrust_mag * 0.68)
        accel[1] -= DRONE_GRAVITY

        # Dynamic weather wind + gusts
        gust_x = math.sin(self.sim_time * 1.7 + float(self.drone_pos[2]) * 0.03) * w.gust_strength
        gust_y = math.cos(self.sim_time * 2.3) * (w.gust_strength * 0.28)
        gust_z = math.cos(self.sim_time * 1.3) * (w.gust_strength * 0.5)
        wind_vec = np.array(
            [w.wind_vec[0] + gust_x, w.wind_vec[1] + gust_y, w.wind_vec[2] + gust_z],
            dtype=np.float32,
        )

        rel_vel = self.drone_vel - wind_vec * 0.35
        accel -= rel_vel * DRONE_DRAG

        self.drone_vel += accel * dt
        spd = float(np.linalg.norm(self.drone_vel))
        if spd > max_spd:
            self.drone_vel *= max_spd / spd

        self.drone_pos += self.drone_vel * dt
        # Keep within broader Kerch Strait arena bounds
        self.drone_pos[0] = clamp(float(self.drone_pos[0]), -220.0, 220.0)
        self.drone_pos[1] = clamp(float(self.drone_pos[1]), -1.0, 165.0)
        self.drone_pos[2] = clamp(float(self.drone_pos[2]), -860.0, 860.0)

        # Battery consumption
        drain = w.drone_battery_drain_mult * (1.55 if inp.boost else 1.0)
        self.drone_battery -= drain * dt
        if self.drone_battery <= 0.0:
            self._destroy_drone_in_air("Села батарея FPV дрона!")
            return

        # Warhead re-arm timer for stand-off airbursts
        self.drone_detonate_cd = max(0.0, self.drone_detonate_cd - dt)

        # Manual stand-off airburst detonation (Space / R2 / Button):
        # the warhead is ejected, the FPV survives and can immediately go for another pass.
        if inp.detonate and self.drone_detonate_cd <= 0.0:
            self._detonate_warhead("Дистанционный подрыв БЧ!", manual=True)
            return

        # Direct / Proximity hit on Fuel Truck hitbox
        truck_center = self.truck_pos + np.array([0.0, 2.15, -0.3], dtype=np.float32)
        # Transform drone position into truck local frame to test accurate rectangular tank + cab box
        rel = self.drone_pos - truck_center
        cy, sy = math.cos(-self.truck_yaw), math.sin(-self.truck_yaw)
        lx = rel[0] * cy + rel[2] * sy
        ly = rel[1]
        lz = -rel[0] * sy + rel[2] * cy
        if abs(lx) <= 2.15 and abs(ly) <= 2.25 and abs(lz) <= 6.1:
            self._detonate_warhead("ПРЯМОЕ ПОПАДАНИЕ FPV В БЕНЗОВОЗ!")
            return

        # Sea or Bridge Deck or Arch Collision
        dx, dy, dz = float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])
        if dy <= 0.35:
            self._destroy_drone_in_air("FPV дрон упал в воды Керченского пролива!")
            return

        # Roadway deck collision
        if abs(dx) <= ROAD_HALF_WIDTH + 0.6 and abs(dz) <= 800.0:
            if ROAD_DECK_Y - 1.8 <= dy <= ROAD_DECK_Y + 0.38:
                self._detonate_warhead("FPV взорвался об полотно моста!")
                return

        # Static barriers & steel shipping arches collision
        for col in self.static_colliders:
            if (col.min_pt[0] <= dx <= col.max_pt[0] and
                    col.min_pt[1] <= dy <= col.max_pt[1] and
                    col.min_pt[2] <= dz <= col.max_pt[2]):
                self._detonate_warhead("FPV врезался в конструкцию моста!")
                return

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

    def _detonate_warhead(self, reason_ru: str, manual: bool = False) -> None:
        if not self.drone_alive:
            return
        frag = self.drone_mod == "airburst_frag"
        shaped = self.drone_mod == "shaped_charge"
        cage = self.truck_mod == "anti_drone_cage"

        blast_radius = DRONE_EXPLOSION_RADIUS * (1.55 if frag else 1.0)
        base_dmg = DRONE_DIRECT_DAMAGE * (1.45 if shaped else 1.0)
        if cage:
            base_dmg *= 0.65

        # Measure distance to closest point on fuel truck cistern/cab
        truck_center = self.truck_pos + np.array([0.0, 2.15, -0.4], dtype=np.float32)
        rel = self.drone_pos - truck_center
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
        dist = float(np.linalg.norm(self.drone_pos - closest_world))

        if dist <= blast_radius:
            falloff = clamp(1.0 - (dist / blast_radius) * 0.78, 0.22, 1.0)
            dealt = base_dmg * falloff
            self.truck_hp = max(0.0, self.truck_hp - dealt)
            # Blast shockwave slows/shakes truck
            self.truck_speed *= 0.72
            reason_ru = f"{reason_ru} (-{int(round(dealt))} HP бензовозу!)"

        self.explosions.append(
            ExplosionFX(
                pos=(float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])),
                radius=blast_radius,
                ttl=1.05,
                max_ttl=1.05,
            )
        )

        if manual:
            # Stand-off charge ejection: the FPV stays in the air and may attack again
            self.drone_detonate_cd = DRONE_DETONATE_COOLDOWN
            if dist > blast_radius:
                reason_ru = f"{reason_ru} (зарряд сорвался вхолостую — БЧ не достала до цистерны!)"
        else:
            # Kamikaze impact: the FPV is always lost
            self.drone_alive = False
            self.drone_lives_left -= 1
            self.drone_respawn_timer = DRONE_RESPAWN_DELAY
        self.last_event_ru = reason_ru

    # -----------------------------------------------------------------------
    # Round Victory / Defeat Evaluation
    # -----------------------------------------------------------------------
    def _check_round_conditions(self) -> None:
        if self.truck_hp <= 0.0:
            # Massive secondary fuel tank fireball!
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
        self.drone_hp = d["hp"]
        self.drone_max_hp = d["max_hp"]
        self.drone_battery = d["battery"]
        self.drone_max_battery = d["max_battery"]
        self.drone_lives_left = d["lives_left"]
        self.drone_alive = d["alive"]
        self.drone_respawn_timer = d["respawn_timer"]
        self.drone_jam_level = d["jam_level"]
        self.drone_detonate_cd = d.get("detonate_cd", 0.0)
        self.drone_acro = bool(d.get("acro", False))

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
