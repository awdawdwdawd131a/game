"""
Smart AI Pilots for Practice Mode & Automated Verification Mode:
- BotDroneController: Flies FPV drone around bridge arches, compensates wind, dives onto Fuel Truck
- BotTruckController: Drives along Crimean Bridge weaving around chicane barriers, aims AA turret & shotgun, triggers EW & smoke when FPV approaches
"""

from __future__ import annotations

import math
import random

import numpy as np

from .config import DRONE_MODIFIERS, ROAD_HALF_WIDTH, TRUCK_MODIFIERS
from .math3d import clamp, wrap_angle
from .physics import DroneInput, TruckInput, WorldSimulation


class BotDroneController:
    def __init__(self, difficulty: float = 0.75) -> None:
        self.difficulty = clamp(difficulty, 0.2, 1.0)
        self.rng = random.Random(101)
        self._weave_phase = 0.0
        self._retreat_timer = 0.0

    def compute_input(self, sim: WorldSimulation, dt: float) -> DroneInput:
        inp = DroneInput()
        if sim.phase == "MODIFIER_DRAFT":
            # Pick a random modifier on round start and mark ready
            idx = (sim.round_number * 2 + 1) % len(DRONE_MODIFIERS)
            inp.pick_modifier = DRONE_MODIFIERS[idx].id
            inp.ready = True
            return inp

        if sim.phase == "MATCH_OVER":
            inp.ready = True
            return inp

        if not sim.drone_alive:
            return inp

        self._weave_phase += dt * 2.4
        self._retreat_timer = max(0.0, self._retreat_timer - dt)

        # Lead the moving fuel truck
        dist = float(np.linalg.norm(sim.truck_pos - sim.drone_pos))
        lead_t = clamp(dist / 48.0, 0.05, 1.4)
        target = sim.truck_pos + sim.truck_vel * lead_t + np.array([0.0, 2.2, 0.0], dtype=np.float32)

        if self._retreat_timer > 0.0:
            # After the airburst: break away upwards and backwards, then come back in
            target = sim.truck_pos + np.array([0.0, 46.0, 62.0], dtype=np.float32)
            target[0] += math.sin(self._weave_phase) * 6.0

        # Keep safe altitude until within terminal dive range (<85m)
        if dist > 85.0:
            target[1] = max(float(target[1]) + 14.0, 34.0)
            target[0] += math.sin(self._weave_phase) * 9.0 * self.difficulty
        elif dist > 26.0:
            # Glide in above the tanker and line up the shot
            target[1] = max(float(target[1]) + 5.0, float(sim.truck_pos[1]) + 6.0)
        else:
            # Terminal run: stay a couple of meters above the cistern, never dive into the deck
            target[1] = max(float(target[1]), float(sim.truck_pos[1]) + 3.4)

        # Avoid bridge structures (steel arches, pylons): push away from the closest point
        # of every nearby AABB and climb over it if the drone is below the top of the structure.
        avoid_x = 0.0
        avoid_z = 0.0
        avoid_y = 0.0
        for col in sim.static_colliders:
            px = clamp(float(sim.drone_pos[0]), float(col.min_pt[0]), float(col.max_pt[0]))
            pz = clamp(float(sim.drone_pos[2]), float(col.min_pt[2]), float(col.max_pt[2]))
            dx = float(sim.drone_pos[0]) - px
            dz = float(sim.drone_pos[2]) - pz
            d2 = dx * dx + dz * dz
            if d2 > 1600.0:
                continue
            d = math.sqrt(d2) + 1e-3
            away = 34.0 / d
            avoid_x += (dx / d) * away
            avoid_z += (dz / d) * away
            # If the drone is inside the footprint of the structure, climb over it
            if (
                float(col.min_pt[0]) <= float(sim.drone_pos[0]) <= float(col.max_pt[0])
                and float(col.min_pt[2]) <= float(sim.drone_pos[2]) <= float(col.max_pt[2])
            ):
                top = float(col.max_pt[1]) + 8.0
                avoid_y = max(avoid_y, top - float(sim.drone_pos[1]))
        if avoid_y > 0.0:
            target[1] = max(float(target[1]), float(sim.drone_pos[1]) + avoid_y)
        # Do not push the drone away from the tanker during the terminal attack
        if dist > 30.0:
            target[0] += clamp(avoid_x, -24.0, 24.0)
            target[2] += clamp(avoid_z, -24.0, 24.0)

        delta = target - sim.drone_pos
        horiz = max(math.hypot(float(delta[0]), float(delta[2])), 1e-4)
        desired_yaw = math.atan2(float(delta[0]), float(delta[2]))
        desired_pitch = math.atan2(-float(delta[1]), horiz)

        yaw_err = wrap_angle(desired_yaw - sim.drone_yaw)
        pitch_err = desired_pitch - sim.drone_pitch

        turn_rate = 3.2 * (0.65 + 0.5 * self.difficulty)
        inp.yaw_delta = clamp(yaw_err, -turn_rate * dt, turn_rate * dt)
        inp.pitch_delta = clamp(pitch_err, -turn_rate * dt, turn_rate * dt)

        if dist > 26.0:
            inp.forward_Axis = 1.0 if abs(yaw_err) < 0.95 else 0.35
            inp.strafe_axis = math.cos(self._weave_phase * 1.3) * 0.45 if dist > 35.0 else 0.0
            inp.boost = dist < 95.0 and abs(yaw_err) < 0.28
        else:
            # Terminal attack: ease off, aim straight at the cistern, hold position
            inp.forward_Axis = clamp(horiz / 30.0, 0.0, 0.55)
            inp.strafe_axis = 0.0
            inp.boost = False
        inp.throttle = clamp((float(target[1]) - float(sim.drone_pos[1])) * 0.08, -0.6, 0.75)

        if sim.drone_mod == "airburst_frag" and dist < 8.0:
            inp.detonate = True
            self._retreat_timer = 3.0
        elif dist < 9.5 and float(sim.drone_pos[1]) < float(sim.truck_pos[1]) + 9.0:
            # Manual airburst like a human pilot pressing the detonation button
            inp.detonate = True
            self._retreat_timer = 3.0

        return inp


class BotTruckController:
    def __init__(self, difficulty: float = 0.75) -> None:
        self.difficulty = clamp(difficulty, 0.2, 1.0)
        self.rng = random.Random(202)
        self._shot_timer = 0.0

    def compute_input(self, sim: WorldSimulation, dt: float) -> TruckInput:
        inp = TruckInput()
        if sim.phase == "MODIFIER_DRAFT":
            idx = (sim.round_number * 3) % len(TRUCK_MODIFIERS)
            inp.pick_modifier = TRUCK_MODIFIERS[idx].id
            inp.ready = True
            return inp

        if sim.phase == "MATCH_OVER":
            inp.ready = True
            return inp

        # 1. Driving & Barrier Avoidance along +Z
        tx, tz = float(sim.truck_pos[0]), float(sim.truck_pos[2])
        desired_x = 0.0
        # Look ahead 55m for closest concrete barrier
        nearest_b = None
        nearest_dz = 999.0
        for b in sim.barriers:
            bz = 0.5 * float(b.min_pt[2] + b.max_pt[2])
            dz = bz - tz
            if -4.0 < dz < 62.0 and dz < nearest_dz:
                nearest_dz = dz
                nearest_b = b

        if nearest_b is not None:
            bx = 0.5 * float(nearest_b.min_pt[0] + nearest_b.max_pt[0])
            # Steer to the opposite half of the highway
            desired_x = -6.8 if bx >= 0.0 else 6.8
        else:
            desired_x = math.sin(sim.sim_time * 0.45) * 3.5

        desired_x = clamp(desired_x, -(ROAD_HALF_WIDTH - 3.5), ROAD_HALF_WIDTH - 3.5)
        lat_err = desired_x - tx
        desired_yaw = clamp(lat_err * 0.085, -0.42, 0.42)
        yaw_err = wrap_angle(desired_yaw - sim.truck_yaw)

        inp.accel = 1.0
        inp.steer = clamp(yaw_err * 3.4, -1.0, 1.0)

        # 2. Gunner Anti-FPV Tracking & Countermeasures
        if sim.drone_alive:
            muzzle = sim.get_turret_muzzle_pos()
            rel = sim.drone_pos + sim.drone_vel * (0.06 * self.difficulty) - muzzle
            dist = float(np.linalg.norm(rel))
            horiz = max(math.hypot(float(rel[0]), float(rel[2])), 1e-4)

            # Slight human-like aim imperfection inversely proportional to difficulty
            wobble = (1.0 - self.difficulty) * 0.14
            aim_yaw = math.atan2(float(rel[0]), float(rel[2])) + math.sin(sim.sim_time * 4.1) * wobble
            aim_pitch = math.atan2(-float(rel[1]), horiz) + math.cos(sim.sim_time * 3.7) * wobble * 0.7

            dyaw = wrap_angle(aim_yaw - sim.turret_yaw)
            dpitch = aim_pitch - sim.turret_pitch
            slew = (2.2 + 2.4 * self.difficulty) * dt
            inp.aim_yaw_delta = clamp(dyaw, -slew, slew)
            inp.aim_pitch_delta = clamp(dpitch, -slew, slew)

            aligned = abs(dyaw) < 0.16 and abs(dpitch) < 0.16
            if aligned and dist < 235.0 and sim.turret_heat < 0.86:
                inp.fire_turret = True
            if abs(dyaw) < 0.22 and abs(dpitch) < 0.22 and dist < 115.0:
                inp.fire_shotgun = True

            # Activate EW Jammer & Smoke when FPV closes in
            if dist < 82.0 and sim.ew_cd <= 0.0:
                inp.trigger_ew = True
            if dist < 68.0 and sim.smoke_cd <= 0.0:
                inp.trigger_smoke = True

        return inp
