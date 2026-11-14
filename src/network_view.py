"""Presentation-only smoothing and truck aim prediction for network matches.

The server simulation remains authoritative.  This module never writes to it:
only a shallow render view receives interpolated transforms and local aim.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Optional

import numpy as np

from .math3d import clamp, wrap_angle
from .physics import TruckInput, WorldSimulation

# The server publishes at 30 Hz. One snapshot interval of interpolation avoids
# displaying identical poses twice at a 60 Hz render rate.
SNAPSHOT_INTERVAL = 1.0 / 30.0
_MAX_PENDING_INPUTS = 600
_POSITIONS = ("truck_pos", "drone_pos")
_ANGLES = ("truck_yaw", "turret_yaw", "drone_yaw", "drone_roll")
_LINEAR = ("turret_pitch", "drone_pitch")


def _pose(sim: WorldSimulation) -> dict:
    values = {key: getattr(sim, key).copy() for key in _POSITIONS}
    values.update({key: float(getattr(sim, key)) for key in _ANGLES + _LINEAR})
    return values


def _blend(start: dict, end: dict, amount: float) -> dict:
    result = {key: start[key] + (end[key] - start[key]) * amount for key in _POSITIONS + _LINEAR}
    for key in _ANGLES:
        result[key] = wrap_angle(start[key] + wrap_angle(end[key] - start[key]) * amount)
    return result


class NetworkView:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start: Optional[dict] = None
        self._target: Optional[dict] = None
        self._display: Optional[dict] = None
        self._elapsed = 0.0
        self._phase_key = None
        self._drone_alive = None
        self._drone_lives = None
        self._pending = deque()
        self._can_predict = False
        self._overflow_seq = 0
        self._latest_seq = 0
        self._aim_yaw = 0.0
        self._aim_pitch = 0.0

    def on_snapshot(self, sim: WorldSimulation, snapshot: dict, role: str) -> None:
        pose = _pose(sim)
        phase_key = (sim.phase, sim.round_number, sim.bridge_route)
        changed = self._phase_key != phase_key
        if changed or self._display is None:
            self._start = _pose(sim)
            self._display = _pose(sim)
            self._pending.clear()
            self._overflow_seq = 0
            self._latest_seq = 0
        else:
            self._start = dict(self._display)
            # Respawns or teleports must not fly through the scenery.
            if (self._drone_alive != sim.drone_alive or self._drone_lives != sim.drone_lives_left
                    or np.linalg.norm(pose["drone_pos"] - self._start["drone_pos"]) > 60):
                self._start["drone_pos"] = pose["drone_pos"].copy()
                for key in ("drone_yaw", "drone_pitch", "drone_roll"):
                    self._start[key] = pose[key]
            if np.linalg.norm(pose["truck_pos"] - self._start["truck_pos"]) > 30:
                self._start["truck_pos"] = pose["truck_pos"].copy()
                self._start["truck_yaw"] = pose["truck_yaw"]
        self._target = pose
        self._elapsed = 0.0
        self._phase_key = phase_key
        self._drone_alive = sim.drone_alive
        self._drone_lives = sim.drone_lives_left

        ack_map = snapshot.get("input_ack")
        ack = ack_map.get("truck") if isinstance(ack_map, dict) else None
        self._can_predict = role == "truck" and sim.phase == "ACTIVE_ROUND" and isinstance(ack, int)
        if self._can_predict:
            while self._pending and self._pending[0][0] <= ack:
                self._pending.popleft()
            if self._overflow_seq:
                if ack >= self._latest_seq:
                    self._overflow_seq = 0
                else:
                    self._can_predict = False
            if self._can_predict:
                self._aim_yaw, self._aim_pitch = sim.turret_yaw, sim.turret_pitch
                for _, yaw, pitch in self._pending:
                    self._aim_yaw = wrap_angle(self._aim_yaw + yaw)
                    self._aim_pitch = clamp(self._aim_pitch + pitch, -1.48, 0.28)
        else:
            self._pending.clear()

    def advance(self, dt: float) -> None:
        if self._target is None or self._start is None:
            return
        self._elapsed += max(0.0, dt)
        self._display = _blend(self._start, self._target, min(1.0, self._elapsed / SNAPSHOT_INTERVAL))

    def truck_input(self, seq: Optional[int], inp: TruckInput) -> None:
        if seq is None:
            return
        self._latest_seq = seq
        if not self._can_predict or self._overflow_seq:
            return
        if len(self._pending) >= _MAX_PENDING_INPUTS:
            # On an extremely stalled link, stop predicting rather than keep an
            # unbounded input history. Resume once the server catches up.
            self._overflow_seq = seq
            self._pending.clear()
            self._can_predict = False
            return
        self._pending.append((seq, inp.aim_yaw_delta, inp.aim_pitch_delta))
        self._aim_yaw = wrap_angle(self._aim_yaw + inp.aim_yaw_delta)
        self._aim_pitch = clamp(self._aim_pitch + inp.aim_pitch_delta, -1.48, 0.28)

    def render_sim(self, sim: WorldSimulation) -> WorldSimulation:
        if sim.phase != "ACTIVE_ROUND" or self._display is None:
            return sim
        view = copy.copy(sim)
        for key, value in self._display.items():
            setattr(view, key, value)
        if self._can_predict:
            view.turret_yaw = self._aim_yaw
            view.turret_pitch = self._aim_pitch
        return view
