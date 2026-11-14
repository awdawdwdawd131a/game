"""Network rendering stays smooth while the authoritative game remains untouched."""
import math
import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from src.app import DesktopGameApp
from src.math3d import wrap_angle
from src.network import GameClient, GameServer, _ClientSender, HEADER_STRUCT
from src.network_view import NetworkView, SNAPSHOT_INTERVAL
from src.physics import DroneInput, TruckInput, WorldSimulation
from src.renderer import SceneRenderer3D


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    return None


class NetworkSmoothingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 520))

    def active(self):
        sim = WorldSimulation(seed=123)
        sim.begin_active_round()
        return sim

    def test_30hz_snapshots_blend_across_60hz_frames_without_mutating_sim(self):
        sim = self.active()
        view = NetworkView()
        sim.truck_pos[:] = [0, 18, 0]
        sim.drone_pos[:] = [0, 24, 100]
        sim.drone_yaw = 0
        view.on_snapshot(sim, {}, "drone")
        sim.truck_pos[2] = 2
        sim.drone_pos[2] = 104
        sim.turret_yaw = .2
        sim.drone_yaw = .4
        view.on_snapshot(sim, {}, "drone")
        view.advance(1 / 60)
        rendered = view.render_sim(sim)
        self.assertAlmostEqual(float(rendered.truck_pos[2]), 1)
        self.assertAlmostEqual(float(rendered.drone_pos[2]), 102)
        self.assertAlmostEqual(rendered.turret_yaw, .1)
        self.assertAlmostEqual(rendered.drone_yaw, .2)
        self.assertEqual(float(sim.truck_pos[2]), 2)
        self.assertEqual(sim.turret_yaw, .2)
        self.assertIsNot(rendered, sim)
        view.advance(1 / 60)
        self.assertAlmostEqual(float(view.render_sim(sim).drone_pos[2]), 104)

    def test_aim_prediction_immediate_and_reconciled_at_350ms(self):
        sim = self.active()
        view = NetworkView()
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        expected = sim.turret_yaw
        # Eleven frames of mouse input while a server round trip is in flight.
        for seq in range(1, 12):
            view.truck_input(seq, TruckInput(aim_yaw_delta=.05, aim_pitch_delta=-.01))
            expected += .05
            self.assertAlmostEqual(view.render_sim(sim).turret_yaw, expected)
        self.assertEqual(sim.turret_yaw, 0)  # server state is not predicted
        sim.turret_yaw = .25
        sim.turret_pitch = -.05
        view.on_snapshot(sim, {"input_ack": {"truck": 5}}, "truck")
        self.assertAlmostEqual(view.render_sim(sim).turret_yaw, .55)
        self.assertAlmostEqual(view.render_sim(sim).turret_pitch, -.11)
        sim.turret_yaw, sim.turret_pitch = .55, -.11
        view.on_snapshot(sim, {"input_ack": {"truck": 11}}, "truck")
        self.assertAlmostEqual(view.render_sim(sim).turret_yaw, .55)
        self.assertEqual(len(view._pending), 0)

    def test_wrap_pitch_clamp_and_respawn_without_flying_across_map(self):
        sim = self.active()
        sim.drone_yaw = math.pi - .1
        view = NetworkView()
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        sim.drone_yaw = -math.pi + .1
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        view.advance(SNAPSHOT_INTERVAL / 2)
        self.assertAlmostEqual(abs(view.render_sim(sim).drone_yaw), math.pi)
        view.truck_input(1, TruckInput(aim_yaw_delta=math.pi + .2, aim_pitch_delta=-9))
        self.assertAlmostEqual(view.render_sim(sim).turret_yaw, wrap_angle(math.pi + .2))
        self.assertEqual(view.render_sim(sim).turret_pitch, -1.48)
        sim.drone_alive = False
        sim.drone_pos[:] = [100, 35, -400]
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        view.advance(.001)
        self.assertTrue(np.allclose(view.render_sim(sim).drone_pos, sim.drone_pos))
        sim.drone_alive = True
        sim.drone_lives_left -= 1
        sim.drone_pos[:] = [0, 30, 300]
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        view.advance(.001)
        self.assertTrue(np.allclose(view.render_sim(sim).drone_pos, sim.drone_pos))

    def test_old_server_round_change_and_delayed_ack_fallback(self):
        sim = self.active()
        view = NetworkView()
        view.on_snapshot(sim, {}, "truck")
        view.truck_input(1, TruckInput(aim_yaw_delta=1))
        self.assertEqual(view.render_sim(sim).turret_yaw, sim.turret_yaw)
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        view.truck_input(2, TruckInput(aim_yaw_delta=1))
        sim.round_number += 1
        sim.start_modifier_draft()
        view.on_snapshot(sim, {"input_ack": {"truck": 2}}, "truck")
        self.assertIs(view.render_sim(sim), sim)
        sim.begin_active_round()
        view.on_snapshot(sim, {"input_ack": {"truck": 2}}, "truck")
        self.assertEqual(view.render_sim(sim).turret_yaw, sim.turret_yaw)
        self.assertFalse(view._pending)

    def test_overflow_disables_prediction_until_ack_catches_up(self):
        sim = self.active()
        view = NetworkView()
        view.on_snapshot(sim, {"input_ack": {"truck": 0}}, "truck")
        for i in range(1, 603):
            view.truck_input(i, TruckInput(aim_yaw_delta=.001))
        self.assertFalse(view._can_predict)
        self.assertFalse(view._pending)
        view.on_snapshot(sim, {"input_ack": {"truck": 10}}, "truck")
        self.assertFalse(view._can_predict)
        view.on_snapshot(sim, {"input_ack": {"truck": 602}}, "truck")
        self.assertTrue(view._can_predict)
        view.truck_input(603, TruckInput(aim_yaw_delta=.1))
        self.assertAlmostEqual(view.render_sim(sim).turret_yaw, .1)

    def test_tcp_ack_is_only_sent_for_consumed_input_and_reconciles(self):
        server = GameServer(host="127.0.0.1", port=0, seed=20)
        drone, truck = GameClient(), GameClient()
        try:
            port = server.start()
            self.assertTrue(drone.connect("127.0.0.1", port, "drone"))
            self.assertTrue(truck.connect("127.0.0.1", port, "truck"))
            with server._lock:
                server.sim.begin_active_round()
                server.sim.turret_pitch = 0
            self.assertTrue(wait_for(lambda: (truck.pop_latest_snapshot() or {}).get("phase") == "ACTIVE_ROUND"))
            seq = truck.send_truck_input(TruckInput(aim_yaw_delta=.4, aim_pitch_delta=-.2))
            self.assertIsNotNone(seq)
            def acked():
                snap = truck.pop_latest_snapshot()
                return snap if snap and snap.get("input_ack", {}).get("truck", 0) >= seq else None
            snap = wait_for(acked)
            self.assertIsNotNone(snap)
            self.assertAlmostEqual(snap["truck"]["turret_yaw"], .4, places=3)
            self.assertAlmostEqual(snap["truck"]["turret_pitch"], -.2, places=3)
        finally:
            drone.disconnect()
            truck.disconnect()
            server.stop()

    def test_live_input_with_delayed_snapshots_keeps_camera_continuous(self):
        # Real TCP input and server physics, with 350 ms of simulated snapshot
        # delivery delay. Prediction must not double-apply acknowledged inputs.
        server = GameServer(host="127.0.0.1", port=0, seed=30)
        drone, truck = GameClient(), GameClient()
        try:
            port = server.start()
            self.assertTrue(drone.connect("127.0.0.1", port, "drone"))
            self.assertTrue(truck.connect("127.0.0.1", port, "truck"))
            with server._lock:
                server.sim.begin_active_round()
                server.sim.turret_yaw = 0
            first = wait_for(lambda: (snap if (snap := truck.pop_latest_snapshot()) and snap["phase"] == "ACTIVE_ROUND" else None))
            self.assertIsNotNone(first)
            sim = self.active()
            sim.apply_snapshot(first)
            view = NetworkView()
            view.on_snapshot(sim, first, "truck")
            delayed = []
            desired = float(sim.turret_yaw)
            deadline = time.monotonic()
            for _ in range(50):
                snap = truck.pop_latest_snapshot()
                if snap:
                    delayed.append((time.monotonic() + .35, snap))
                now = time.monotonic()
                while delayed and delayed[0][0] <= now:
                    _, old = delayed.pop(0)
                    sim.apply_snapshot(old)
                    view.on_snapshot(sim, old, "truck")
                inp = TruckInput(aim_yaw_delta=.01)
                seq = truck.send_truck_input(inp)
                view.advance(1 / 60)
                view.truck_input(seq, inp)
                desired += .01
                self.assertAlmostEqual(view.render_sim(sim).turret_yaw, desired, delta=.002)
                deadline += 1 / 60
                time.sleep(max(0, deadline - time.monotonic()))
            final_seq = seq
            def last_ack():
                snap = truck.pop_latest_snapshot()
                return snap if snap and snap.get("input_ack", {}).get("truck", 0) >= final_seq else None
            snap = wait_for(last_ack)
            self.assertIsNotNone(snap)
            self.assertAlmostEqual(snap["truck"]["turret_yaw"], desired, delta=.003)
            sim.apply_snapshot(snap)
            view.on_snapshot(sim, snap, "truck")
            self.assertAlmostEqual(view.render_sim(sim).turret_yaw, desired, delta=.003)
            self.assertFalse(view._pending)
        finally:
            drone.disconnect()
            truck.disconnect()
            server.stop()

    def test_client_coalescing_preserves_newest_sequence(self):
        class Sock:
            def __init__(self):
                self.frames = []
            def sendall(self, frame):
                self.frames.append(frame)
            def shutdown(self, how):
                pass
            def close(self):
                pass
        import json
        sock = Sock()
        sender = _ClientSender(sock)
        sender.input(TruckInput(aim_yaw_delta=.1, fire_shotgun=True).to_dict(), 1)
        sender.input(TruckInput(aim_yaw_delta=.2).to_dict(), 2)
        sender.start()
        self.assertTrue(wait_for(lambda: bool(sock.frames)))
        sender.close()
        pkt = json.loads(sock.frames[0][HEADER_STRUCT.size:])
        self.assertEqual(pkt["seq"], 2)
        self.assertAlmostEqual(pkt["data"]["aim_yaw_delta"], .3)
        self.assertTrue(pkt["data"]["fire_shotgun"])

    def test_app_network_truck_camera_moves_every_frame_before_ack(self):
        app = DesktopGameApp(os.getcwd(), 800, 520)
        sim = self.active()
        class DelayedClient:
            connected = True
            error_msg = ""
            online_roles = ["truck"]
            ping_ms = 350
            def __init__(self):
                self.seq = 0
                self.snap = sim.get_snapshot()
                self.snap["input_ack"] = {"truck": 0}
            def pop_latest_snapshot(self):
                snap, self.snap = self.snap, None
                return snap
            def send_truck_input(self, inp):
                self.seq += 1
                return self.seq
            def cancel_connect(self):
                pass
            def disconnect(self):
                self.connected = False
        try:
            app.sim = self.active()
            app.state, app.my_role = "GAME_NET", "truck"
            app.net_client = DelayedClient()
            app._net_has_snapshot = False
            app._trigger_audio_cues = lambda: None
            app._update_ambient_audio = lambda: None
            with patch.object(app.controls, "build_truck_input", return_value=TruckInput(aim_yaw_delta=.1)):
                observed = []
                for _ in range(5):
                    app._update(1 / 60)
                    rendered = app._network_view.render_sim(app.sim)
                    observed.append(app.renderer.make_truck_camera(rendered).yaw)
            np.testing.assert_allclose(observed, [.1, .2, .3, .4, .5], atol=1e-6)
            self.assertAlmostEqual(app.sim.turret_yaw, 0.0)
        finally:
            app._cleanup_network()

    def test_full_frame_budget_controls_render_scale(self):
        sim = self.active()
        renderer = SceneRenderer3D(800, 520, sim.barriers, os.getcwd())
        for _ in range(10):
            renderer.report_frame_time(25.0)  # 40 FPS due to update / networking
            renderer._update_render_scale(10.0)  # renderer alone looks fast
        self.assertLess(renderer.render_scale, 1.0)
        self.assertGreaterEqual(renderer.render_scale, .5)
        for _ in range(120):
            renderer.report_frame_time(16.0)
            renderer._update_render_scale(10.0)
        self.assertGreater(renderer.render_scale, .5)


if __name__ == "__main__":
    unittest.main()
