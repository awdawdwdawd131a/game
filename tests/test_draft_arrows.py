"""Arrow-key draft navigation, including delayed authoritative snapshots."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy
import unittest
from collections import defaultdict
from unittest.mock import patch

import pygame

from src.app import DesktopGameApp
from src.controls import InputController
from src.physics import WorldSimulation


class DraftArrowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 520))

    def make_app(self):
        app = DesktopGameApp(os.getcwd(), 800, 520)
        pygame.event.clear()
        self.addCleanup(app._cleanup_network)
        return app

    def post_keys(self, app, *keys):
        for key in keys:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        app._process_events()

    def test_arrows_wrap_and_only_change_draft(self):
        ctrl = InputController()
        options = ["a", "b", "c"]
        ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP), draft_mode=True)
        self.assertEqual(ctrl.build_drone_input(1 / 60, False, options).pick_modifier, "c")
        ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN), draft_mode=True)
        self.assertEqual(ctrl.build_drone_input(1 / 60, False, options).pick_modifier, "a")
        ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN), draft_mode=False)
        self.assertEqual(ctrl.build_drone_input(1 / 60, False, options).pick_modifier, "")
        keys = defaultdict(bool, {pygame.K_UP: True})
        with patch("pygame.key.get_pressed", return_value=keys):
            self.assertEqual(ctrl.build_drone_input(1 / 60, False).forward_Axis, 1.0)
            self.assertEqual(ctrl.build_truck_input(1 / 60, False).accel, 1.0)

    def test_mixed_number_gamepad_and_arrows(self):
        ctrl = InputController()
        options = ["a", "b", "c"]
        ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3), draft_mode=True)
        ctrl.handle_event(pygame.event.Event(pygame.JOYHATMOTION, value=(0, -1), joy=0, hat=0), draft_mode=True)
        self.assertEqual(ctrl.build_truck_input(1 / 60, False, options).pick_modifier, "a")
        ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP), draft_mode=True)
        self.assertEqual(ctrl.build_truck_input(1 / 60, False, options).pick_modifier, "c")

    def test_local_both_roles_arrows_and_enter(self):
        for role in ("drone", "truck"):
            with self.subTest(role=role):
                app = self.make_app()
                app._start_local_practice(role)
                options = list(app.sim.drone_draft_options if role == "drone" else app.sim.truck_draft_options)
                self.post_keys(app, pygame.K_DOWN, pygame.K_DOWN, pygame.K_RETURN)
                app._update(1 / 60)
                self.assertEqual(app.sim.phase, "ACTIVE_ROUND")
                chosen = app.sim.drone_mod if role == "drone" else app.sim.truck_mod
                self.assertEqual(chosen, options[2])
                app._cleanup_network()

    def test_delayed_snapshots_do_not_reset_cursor_and_new_round_does(self):
        app = self.make_app()
        sim = WorldSimulation(seed=42)
        first = sim.get_snapshot()
        options = first["drone_draft_options"]
        class DelayedClient:
            connected = True
            error_msg = ""
            online_roles = ["drone"]
            ping_ms = 0.0
            def __init__(self, snapshot):
                self.snapshot = snapshot
                self.sent = []
            def pop_latest_snapshot(self):
                return copy.deepcopy(self.snapshot)
            def send_drone_input(self, inp):
                self.sent.append(inp)
            def cancel_connect(self):
                pass
            def disconnect(self):
                self.connected = False
        client = DelayedClient(first)
        app.net_client = client
        app.state = "GAME_NET"
        app.my_role = "drone"
        app.sim = WorldSimulation(seed=9)
        app.sim.apply_snapshot(first)
        app._net_has_snapshot = True
        app._draft_navigation_key = None
        self.post_keys(app, pygame.K_DOWN)
        app._update(1 / 60)
        self.assertEqual(client.sent[-1].pick_modifier, options[1])
        # The next server update is deliberately stale; the next press must
        # advance from our local selection rather than snap back to option 1.
        self.post_keys(app, pygame.K_DOWN)
        app._update(1 / 60)
        self.assertEqual(client.sent[-1].pick_modifier, options[2])
        with patch.object(app.renderer, "draw_phase_overlay", wraps=app.renderer.draw_phase_overlay) as overlay:
            app._render_gameplay()
        self.assertEqual(overlay.call_args.kwargs["draft_choice"], options[2])
        sim.round_number = 2
        sim.start_modifier_draft()
        client.snapshot = sim.get_snapshot()
        app._update(1 / 60)
        self.post_keys(app, pygame.K_UP)
        app._update(1 / 60)
        self.assertEqual(client.sent[-1].pick_modifier, sim.drone_draft_options[2])

    def test_arrows_before_first_network_snapshot_are_kept(self):
        app = self.make_app()
        app.preferred_role = "drone"
        app.host_bot_fill = False
        app._start_auto_host_setup = lambda port: None
        app.controls._draft_cursor = 2  # cursor from an earlier match
        app._launch_hosted_match(0)
        self.assertEqual(app.controls._draft_cursor, 0)
        self.assertFalse(app._net_has_snapshot)
        self.post_keys(app, pygame.K_DOWN, pygame.K_DOWN, pygame.K_RETURN)
        import time
        deadline = time.monotonic() + 4
        ready = False
        while time.monotonic() < deadline:
            app._update(1 / 60)
            with app.hosted_server._lock:
                server_sim = app.hosted_server.sim
                ready = server_sim.drone_draft_ready
                chosen = server_sim.drone_mod
                options = list(server_sim.drone_draft_options)
            if ready:
                break
            time.sleep(.02)
        self.assertTrue(ready)
        self.assertEqual(chosen, options[2])


if __name__ == "__main__":
    unittest.main()
