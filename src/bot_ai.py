"""
Smart AI Pilots for Practice Mode & Automated Verification Mode:
- BotDroneController: Flies FPV drone in Acro dynamics around bridge arches, compensates wind/gravity, dives onto Fuel Truck
- BotTruckController: Drives along Crimean Bridge weaving around chicane barriers, leads targets with 3D ballistics, triggers EW & smoke
"""

from __future__ import annotations

import math
import random

import numpy as np

from .config import ROAD_HALF_WIDTH
from .math3d import clamp, wrap_angle
from .physics import DroneInput, TruckInput, WorldSimulation


class BotDroneController:
    def __init__(self, difficulty: float = 0.75) -> None:
        self.difficulty = clamp(difficulty, 0.2, 1.0)
        self.rng = random.Random(101)
        self._weave_phase = 0.0
        self._retreat_timer = 0.0

    def compute_input(self, sim: WorldSimulation, dt: float) -> DroneInput:
        inp = DroneInput(acro_mode=True)
        if sim.phase == "MODIFIER_DRAFT":
            inp.ready = True
            return inp

        if sim.phase == "MATCH_OVER":
            inp.ready = True
            return inp

        if not sim.drone_alive:
            return inp

        self._weave_phase += dt * 2.4
        self._retreat_timer = max(0.0, self._retreat_timer - dt)

        # Lead the moving fuel truck using actual closing speed
        dist = float(np.linalg.norm(sim.truck_pos - sim.drone_pos))
        closing_spd = max(38.0, float(np.linalg.norm(sim.drone_vel - sim.truck_vel)))
        lead_t = clamp(dist / closing_spd, 0.02, 0.95)
        target = sim.truck_pos + sim.truck_vel * lead_t + np.array([0.0, 2.0, 0.0], dtype=np.float32)

        if self._retreat_timer > 0.0:
            target = sim.truck_pos + np.array([0.0, 44.0, 58.0], dtype=np.float32)
            target[0] += math.sin(self._weave_phase) * 6.0

        # Smooth realistic FPV glide-dive profile starting from 240m out
        if dist > 35.0:
            dive_alt = clamp((dist - 35.0) * 0.085, 0.0, 20.0)
            weave_amp = clamp((dist - 105.0) / 100.0, 0.0, 1.0) * 5.0 * self.difficulty
            target[1] = float(sim.truck_pos[1]) + 2.0 + dive_alt
            target[0] += math.sin(self._weave_phase) * weave_amp
        else:
            target[1] = float(sim.truck_pos[1]) + 1.8

        # Stay cleanly inside the 26m highway corridor when passing through the bridge arches
        if abs(float(sim.drone_pos[2])) < 145.0 or abs(float(target[2])) < 145.0:
            target[0] = clamp(float(target[0]), -7.5, 7.5)
            if abs(float(sim.drone_pos[0])) > 10.0 and float(sim.drone_pos[1]) < 48.0:
                target[1] = max(float(target[1]), 42.0)

        # Avoid bridge structures at our altitude (ignore low road barriers when flying well above them!)
        avoid_x = 0.0
        avoid_z = 0.0
        avoid_y = 0.0
        dy_drone = float(sim.drone_pos[1])
        for col in sim.static_colliders:
            if dy_drone > float(col.max_pt[1]) + 2.5 or dy_drone < float(col.min_pt[1]) - 3.0:
                continue
            px = clamp(float(sim.drone_pos[0]), float(col.min_pt[0]), float(col.max_pt[0]))
            pz = clamp(float(sim.drone_pos[2]), float(col.min_pt[2]), float(col.max_pt[2]))
            dx = float(sim.drone_pos[0]) - px
            dz = float(sim.drone_pos[2]) - pz
            d2 = dx * dx + dz * dz
            if d2 > 900.0:
                continue
            d = math.sqrt(d2) + 1e-3
            away = 22.0 / d
            avoid_x += (dx / d) * away
            avoid_z += (dz / d) * away
            if (
                float(col.min_pt[0]) <= float(sim.drone_pos[0]) <= float(col.max_pt[0])
                and float(col.min_pt[2]) <= float(sim.drone_pos[2]) <= float(col.max_pt[2])
            ):
                top = float(col.max_pt[1]) + 6.0
                avoid_y = max(avoid_y, top - dy_drone)
        if avoid_y > 0.0:
            target[1] = max(float(target[1]), dy_drone + avoid_y)
        if dist > 40.0:
            target[0] += clamp(avoid_x, -16.0, 16.0)
            target[2] += clamp(avoid_z, -16.0, 16.0)

        delta = target - sim.drone_pos
        dist_t = max(float(np.linalg.norm(delta)), 1e-3)
        # Velocity-pursuit guidance (crabs into crosswinds and cancels lateral drift!)
        desired_spd = 46.0 if dist > 90.0 else 38.0
        desired_vel = (delta / dist_t) * desired_spd
        vel_corr = desired_vel - sim.drone_vel
        aim_vec = delta + vel_corr * 1.15
        aim_vec[1] = delta[1]  # keep explicit altitude/dive control below

        horiz = max(math.hypot(float(aim_vec[0]), float(aim_vec[2])), 1e-4)
        desired_yaw = math.atan2(float(aim_vec[0]), float(aim_vec[2]))
        desired_pitch = clamp(math.atan2(-float(delta[1]), max(math.hypot(float(delta[0]), float(delta[2])), 1e-4)), -0.55, 0.90)
        if dist > 75.0:
            desired_pitch = clamp(desired_pitch + 0.10, -0.35, 0.70)

        yaw_err = wrap_angle(desired_yaw - sim.drone_yaw)
        pitch_err = desired_pitch - sim.drone_pitch

        turn_rate = 3.8 * (0.65 + 0.5 * self.difficulty)
        inp.yaw_delta = clamp(yaw_err, -turn_rate * dt, turn_rate * dt)

        if dist > 28.0:
            inp.forward_Axis = 1.0 if abs(yaw_err) < 0.85 else 0.35
            inp.boost = dist < 130.0 and abs(yaw_err) < 0.38
        else:
            inp.forward_Axis = 0.85 if abs(yaw_err) < 0.65 else 0.30
            inp.boost = dist < 24.0 and abs(yaw_err) < 0.25

        # Compensate cyclic nose-down drift in pitch_delta so bot tracks desired dive angle cleanly
        inp.pitch_delta = clamp(pitch_err - inp.forward_Axis * 0.55 * dt, -turn_rate * dt, turn_rate * dt)

        # Active Acro roll bank control: bank into turns & cancel lateral slip
        weave_bank = math.sin(self._weave_phase * 1.1) * 0.20 if dist > 110.0 else 0.0
        desired_roll = clamp(-yaw_err * 0.75 + weave_bank, -0.65, 0.65)
        inp.strafe_axis = clamp(-(desired_roll - sim.drone_roll) * 2.1, -1.0, 1.0)

        # Active vertical-speed throttle loop to stay airborne in tilted-rotor physics
        desired_vy = clamp((float(target[1]) - float(sim.drone_pos[1])) * 1.45, -18.0, 16.0)
        vy_err = desired_vy - float(sim.drone_vel[1])
        inp.throttle = clamp(0.40 + vy_err * 0.14 + abs(sim.drone_roll) * 0.24, -0.55, 1.0)

        if sim.drone_mod == "cluster_munitions":
            if horiz < 13.0 and 10.0 < (float(sim.drone_pos[1]) - float(sim.truck_pos[1])) < 42.0:
                inp.detonate = True
                self._retreat_timer = 2.6
        elif sim.drone_mod == "airburst_frag" and dist < 6.8:
            inp.detonate = True
            self._retreat_timer = 2.8
        elif dist < 5.5 and float(sim.drone_pos[1]) < float(sim.truck_pos[1]) + 6.0:
            inp.detonate = True
            self._retreat_timer = 2.8

        return inp


class BotTruckController:
    def __init__(self, difficulty: float = 0.75) -> None:
        self.difficulty = clamp(difficulty, 0.2, 1.0)
        self.rng = random.Random(202)

    def compute_input(self, sim: WorldSimulation, dt: float) -> TruckInput:
        inp = TruckInput()
        if sim.phase == "MODIFIER_DRAFT":
            inp.ready = True
            return inp

        if sim.phase == "MATCH_OVER":
            inp.ready = True
            return inp

        # 1. Driving & Barrier Avoidance along +Z
        tx, tz = float(sim.truck_pos[0]), float(sim.truck_pos[2])
        desired_x = 0.0
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
            desired_x = -6.8 if bx >= 0.0 else 6.8
        else:
            desired_x = math.sin(sim.sim_time * 0.45) * 3.5

        desired_x = clamp(desired_x, -(ROAD_HALF_WIDTH - 3.5), ROAD_HALF_WIDTH - 3.5)
        lat_err = desired_x - tx
        desired_yaw = clamp(lat_err * 0.085, -0.42, 0.42)
        yaw_err = wrap_angle(desired_yaw - sim.truck_yaw)

        inp.accel = 1.0
        inp.steer = clamp(yaw_err * 3.4, -1.0, 1.0)

        # 2. Gunner Anti-FPV Tracking with 3D Ballistic Lead Solver & Countermeasures
        if sim.drone_alive:
            muzzle = sim.get_turret_muzzle_pos()
            lead_pos = sim.compute_ballistic_lead_pos()
            raw_target = sim.drone_pos
            aim_target = raw_target + (lead_pos - raw_target) * (0.65 + 0.35 * self.difficulty) if lead_pos is not None else raw_target

            rel = aim_target - muzzle
            dist = float(np.linalg.norm(sim.drone_pos - muzzle))
            horiz = max(math.hypot(float(rel[0]), float(rel[2])), 1e-4)

            wobble = (1.0 - self.difficulty) * 0.42
            aim_yaw = math.atan2(float(rel[0]), float(rel[2])) + math.sin(sim.sim_time * 3.8) * wobble
            aim_pitch = math.atan2(-float(rel[1]), horiz) + math.cos(sim.sim_time * 3.3) * wobble * 0.75

            dyaw = wrap_angle(aim_yaw - sim.turret_yaw)
            dpitch = aim_pitch - sim.turret_pitch
            slew = (1.2 + 1.4 * self.difficulty) * dt
            inp.aim_yaw_delta = clamp(dyaw, -slew, slew)
            inp.aim_pitch_delta = clamp(dpitch, -slew, slew)

            max_engage = 205.0 if sim.truck_mod in ("buckshot_radar", "ballistic_computer") else 165.0
            aligned = abs(dyaw) < 0.14 and abs(dpitch) < 0.14
            if aligned and dist < max_engage and sim.turret_heat < 0.86:
                inp.fire_turret = True
            if abs(dyaw) < 0.22 and abs(dpitch) < 0.22 and dist < 95.0:
                inp.fire_shotgun = True

            # Activate EW Jammer & Smoke when FPV closes in
            if dist < 95.0 and sim.ew_cd <= 0.0:
                inp.trigger_ew = True
            if dist < 68.0 and sim.smoke_cd <= 0.0:
                inp.trigger_smoke = True

        return inp
