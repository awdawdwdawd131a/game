"""Gameplay regressions; run with SDL_VIDEODRIVER=dummy python -m unittest discover -s tests."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pygame

from src.app import DesktopGameApp
from src.bot_ai import BotTruckController
from src.config import DRONE_MODIFIERS, EW_JAMMER_RADIUS, RAIL_CENTER_X, RAIL_DECK_Y, WEATHER_PRESETS
from src.controls import InputController
from src.network import GameClient, GameServer
from src.physics import DroneInput, TruckInput, WorldSimulation
from src.renderer import SceneRenderer3D


class GameplayRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 520))

    def test_controls_and_modifier_selection(self):
        ctrl = InputController()
        self.assertFalse(ctrl.acro_toggle)
        keys = defaultdict(bool, {pygame.K_w: True, pygame.K_LSHIFT: True})
        with patch("pygame.key.get_pressed", return_value=keys):
            inp = ctrl.build_drone_input(1 / 60, False)
        self.assertEqual(inp.forward_Axis, 1.0)
        self.assertEqual(inp.throttle, 1.0)
        options = [m.id for m in DRONE_MODIFIERS[:3]]
        ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        self.assertEqual(ctrl.build_drone_input(1 / 60, False, options).pick_modifier, options[2])
        self.assertFalse(ctrl.build_drone_input(1 / 60, False, options).pick_modifier)
        ctrl.handle_event(pygame.event.Event(pygame.JOYHATMOTION, value=(0, -1), joy=0, hat=0))
        self.assertEqual(ctrl.build_drone_input(1 / 60, False, options).pick_modifier, options[0])

    def test_menu_enter_does_not_auto_confirm_draft(self):
        app = DesktopGameApp.__new__(DesktopGameApp)
        app.controls = InputController()
        app.controls.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        app._cleanup_network = lambda: None
        app._set_mouse_capture = lambda enabled: setattr(app, "mouse_captured", enabled)
        app._start_local_practice("drone")
        self.assertFalse(app.controls.build_drone_input(1 / 60, False).ready)
        self.assertEqual(app.sim.phase, "MODIFIER_DRAFT")

    def test_draft_choice_role_validation_and_ready(self):
        sim = WorldSimulation(seed=3)
        drone_choice, truck_choice = sim.drone_draft_options[1], sim.truck_draft_options[2]
        sim.step(0.01, DroneInput(pick_modifier=drone_choice, ready=True), TruckInput(pick_modifier=truck_choice))
        self.assertEqual(sim.drone_mod, drone_choice)
        self.assertEqual(sim.truck_mod, truck_choice)
        self.assertTrue(sim.drone_draft_ready)
        sim.step(0.01, DroneInput(pick_modifier=truck_choice), TruckInput(ready=True))
        self.assertEqual(sim.phase, "ACTIVE_ROUND")
        self.assertEqual(sim.drone_mod, drone_choice)
        self.assertEqual(sim.truck_mod, truck_choice)

    def test_waiting_lobby_preserves_choice(self):
        server = GameServer(host="127.0.0.1", port=0, fill_with_bot=False, seed=7)
        port = server.start()
        client = GameClient()
        try:
            self.assertTrue(client.connect("127.0.0.1", port, preferred_role="drone", timeout=3))
            with server._lock:
                choice = server.sim.drone_draft_options[1]
            client.send_drone_input(DroneInput(pick_modifier=choice, ready=True))
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with server._lock:
                    picked = server.sim.drone_mod == choice and server.sim.drone_draft_ready
                    timer = server.sim.phase_timer
                if picked:
                    break
                time.sleep(0.02)
            self.assertTrue(picked, "choice while waiting for player two was lost")
            self.assertEqual(timer, 30.0)
        finally:
            client.disconnect()
            server.stop()

    def test_both_bridge_routes_and_snapshot(self):
        sim = WorldSimulation(seed=3)
        sim.round_number = 2
        sim.start_modifier_draft()
        self.assertEqual(sim.bridge_route, "rail")
        sim.begin_active_round()
        self.assertAlmostEqual(float(sim.truck_pos[0]), RAIL_CENTER_X)
        self.assertAlmostEqual(float(sim.truck_pos[1]), RAIL_DECK_Y)
        for _ in range(60):
            sim.step(1 / 60, DroneInput(), TruckInput(accel=1, steer=1))
        self.assertLess(abs(float(sim.truck_pos[0]) - RAIL_CENTER_X), 5)
        self.assertGreater(float(sim.truck_pos[2]), -560)
        copy = WorldSimulation(seed=4)
        copy.apply_snapshot(sim.get_snapshot())
        self.assertEqual(copy.bridge_route, "rail")
        self.assertEqual(float(copy.truck_pos[1]), RAIL_DECK_Y)
        self.assertTrue(any(c.tag == "rail_arch" for c in sim.static_colliders))
        sim.round_number = 3
        sim.start_modifier_draft()
        self.assertEqual(sim.bridge_route, "road")

    def test_jammer_near_far_and_fiber(self):
        sim = WorldSimulation(seed=2)
        sim.drone_mod = "sport_motors"
        sim.truck_mod = "v12_biturbo"
        sim.begin_active_round()
        sim.ew_active_timer = 4
        sim.drone_pos[:] = sim.truck_pos + np.array([0, 2.2, EW_JAMMER_RADIUS + 10])
        sim._step_drone(1 / 60, DroneInput())
        self.assertEqual(sim.drone_jam_level, 0)
        sim.drone_pos[:] = sim.truck_pos + np.array([0, 5, 15])
        sim._step_drone(1 / 60, DroneInput())
        self.assertGreater(sim.drone_jam_level, 0.8)
        sim.drone_mod = "fiber_optic"
        sim._step_drone(1 / 60, DroneInput())
        self.assertEqual(sim.drone_jam_level, 0)

    def test_fpv_hud_has_no_target_marker_and_jam_renders(self):
        sim = WorldSimulation(seed=3)
        renderer = SceneRenderer3D(800, 520, sim.barriers, os.getcwd())
        one = pygame.Surface((800, 520))
        two = pygame.Surface((800, 520))
        renderer.draw_drone_hud(one, sim, (300, 180, 100))
        renderer.draw_drone_hud(two, sim, None)
        self.assertTrue(np.array_equal(pygame.surfarray.array3d(one), pygame.surfarray.array3d(two)))
        sim.drone_jam_level = 0.95
        renderer.draw_drone_hud(two, sim, None)
        self.assertFalse(np.array_equal(pygame.surfarray.array3d(one), pygame.surfarray.array3d(two)))

    def test_escape_and_capture_release(self):
        app = DesktopGameApp.__new__(DesktopGameApp)
        app.state = "GAME_LOCAL"
        app.mouse_captured = True
        app._cleanup_network = lambda: None
        app._set_mouse_capture = lambda enabled: setattr(app, "mouse_captured", enabled)
        app._handle_gameplay_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertEqual(app.state, "MENU")
        self.assertFalse(app.mouse_captured)

    def test_tab_release_is_not_immediately_recaptured(self):
        app = DesktopGameApp.__new__(DesktopGameApp)
        app.state = "GAME_LOCAL"
        app.sim = WorldSimulation(seed=5)
        app.sim.begin_active_round()
        app.my_role = "drone"
        app._last_game_phase = "MODIFIER_DRAFT"
        app.mouse_captured = False
        app.controls = InputController()
        app.bot_truck = BotTruckController(.76)
        app._set_mouse_capture = lambda enabled: setattr(app, "mouse_captured", enabled)
        app._cleanup_network = lambda: None
        app._trigger_audio_cues = lambda: None
        app._update_ambient_audio = lambda: None
        app._update(1 / 60)
        self.assertTrue(app.mouse_captured)
        app._handle_gameplay_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        self.assertFalse(app.mouse_captured)
        app._update(1 / 60)
        self.assertFalse(app.mouse_captured)

    def test_weather_and_bot_reaction(self):
        self.assertGreaterEqual(len(WEATHER_PRESETS), 9)
        self.assertLess(next(w.fog_end for w in WEATHER_PRESETS if w.id == "kerch_fog"), 150)
        sim = WorldSimulation(seed=3)
        sim.begin_active_round()
        sim.drone_pos[:] = sim.truck_pos + np.array([0, 10, 70])
        bot = BotTruckController(.76)
        self.assertFalse(bot.compute_input(sim, 1 / 60).fire_turret)
        self.assertLess(bot._reaction, 0.1)
        sim.drone_alive = False
        bot.compute_input(sim, 1 / 60)
        sim.drone_alive = True
        bot.compute_input(sim, 1 / 60)
        self.assertLess(bot._reaction, 0.1)


if __name__ == "__main__":
    unittest.main()
