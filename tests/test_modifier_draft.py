"""Tests for three-option drafting and each new loadout's gameplay effects."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import unittest
import numpy as np
import pygame

from src.config import (DRONE_MODIFIERS, TRUCK_MODIFIERS, DRONE_BASE_HP,
                        DRONE_BATTERY_SECONDS, DRONE_DIRECT_DAMAGE, DRONE_EXPLOSION_RADIUS,
                        SHOTGUN_COOLDOWN, SHOTGUN_PELLET_DMG, EW_JAMMER_DURATION,
                        EW_JAMMER_COOLDOWN, TRUCK_BASE_HP, TRUCK_BASE_MAX_SPEED)
from src.physics import WorldSimulation, DroneInput, TruckInput
from src.renderer import SceneRenderer3D
from src.network import GameServer, GameClient
from src.app import DesktopGameApp


class ModifierDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 520))

    def sim(self, drone="fiber_optic", truck="v12_biturbo"):
        sim = WorldSimulation(seed=151)
        sim.drone_mod = drone
        sim.truck_mod = truck
        sim.weather_id = "clear_day"
        sim.begin_active_round()
        return sim

    def test_deal_is_three_unique_and_seeded_with_full_pool_coverage(self):
        self.assertGreaterEqual(len(DRONE_MODIFIERS), 15)
        self.assertGreaterEqual(len(TRUCK_MODIFIERS), 15)
        a = WorldSimulation(seed=12)
        b = WorldSimulation(seed=12)
        seen_d, seen_t = set(), set()
        for _ in range(100):
            self.assertEqual(a.drone_draft_options, b.drone_draft_options)
            self.assertEqual(a.truck_draft_options, b.truck_draft_options)
            self.assertEqual(len(set(a.drone_draft_options)), 3)
            self.assertEqual(len(set(a.truck_draft_options)), 3)
            self.assertEqual(a.drone_mod, a.drone_draft_options[0])
            self.assertEqual(a.truck_mod, a.truck_draft_options[0])
            seen_d.update(a.drone_draft_options)
            seen_t.update(a.truck_draft_options)
            a.start_modifier_draft()
            b.start_modifier_draft()
        self.assertEqual(seen_d, {m.id for m in DRONE_MODIFIERS})
        self.assertEqual(seen_t, {m.id for m in TRUCK_MODIFIERS})

    def test_only_offered_choices_and_timeout_default(self):
        s = WorldSimulation(seed=17)
        outside = next(m.id for m in DRONE_MODIFIERS if m.id not in s.drone_draft_options)
        initial = s.drone_mod
        s.step(.01, DroneInput(pick_modifier=outside, ready=True), TruckInput())
        self.assertEqual(s.drone_mod, initial)
        self.assertFalse(s.drone_draft_ready)
        s.step(.01, DroneInput(pick_modifier=s.drone_draft_options[2]), TruckInput(pick_modifier=s.truck_draft_options[1]))
        self.assertEqual(s.drone_mod, s.drone_draft_options[2])
        self.assertEqual(s.truck_mod, s.truck_draft_options[1])
        s.step(31.0, DroneInput(), TruckInput())
        self.assertEqual(s.phase, "ACTIVE_ROUND")
        self.assertEqual(s.drone_mod, s.drone_draft_options[2])

    def test_snapshot_and_live_tcp_choices(self):
        srv = GameServer(host="127.0.0.1", port=0, seed=50)
        port = srv.start()
        c1, c2 = GameClient(), GameClient()
        try:
            self.assertTrue(c1.connect("127.0.0.1", port, preferred_role="drone", timeout=3))
            self.assertTrue(c2.connect("127.0.0.1", port, preferred_role="truck", timeout=3))
            deadline = time.monotonic() + 3
            snap = None
            while time.monotonic() < deadline and snap is None:
                snap = c1.pop_latest_snapshot()
                time.sleep(.01)
            self.assertIsNotNone(snap)
            d_opts, t_opts = snap["drone_draft_options"], snap["truck_draft_options"]
            self.assertEqual(len(d_opts), 3)
            self.assertEqual(len(t_opts), 3)
            view = WorldSimulation(seed=51)
            view.apply_snapshot(snap)
            self.assertEqual(view.drone_draft_options, d_opts)
            self.assertEqual(view.truck_draft_options, t_opts)
            c1.send_drone_input(DroneInput(pick_modifier=d_opts[2], ready=True))
            c2.send_truck_input(TruckInput(pick_modifier=t_opts[1], ready=True))
            active = None
            while time.monotonic() < deadline:
                latest = c2.pop_latest_snapshot()
                if latest and latest["phase"] == "ACTIVE_ROUND":
                    active = latest
                    break
                time.sleep(.01)
            self.assertIsNotNone(active)
            self.assertEqual(active["drone_mod"], d_opts[2])
            self.assertEqual(active["truck_mod"], t_opts[1])
        finally:
            c1.disconnect()
            c2.disconnect()
            srv.stop()

    def test_host_waits_for_first_snapshot_before_resolving_choice(self):
        app = DesktopGameApp(os.getcwd(), 800, 520)
        app.preferred_role = "drone"
        app.host_bot_fill = False
        app._start_auto_host_setup = lambda port: None
        try:
            app._launch_hosted_match(0)
            self.assertEqual(app.state, "GAME_NET")
            self.assertFalse(app._net_has_snapshot)
            self.assertEqual(app.sim.drone_draft_options, [])
            pygame.event.clear()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
            app._process_events()
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
        finally:
            app._cleanup_network()

    def test_quick_reload_and_precision_charge(self):
        quick = self.sim(drone="quick_reload")
        base = self.sim(drone="wind_gyro")
        precise = self.sim(drone="precision_charge")
        for s in (quick, base, precise):
            s.drone_pos[:] = s.truck_pos + np.array([0, 2.15, -.4])
            s._detonate_warhead("test", manual=True)
        self.assertAlmostEqual(quick.drone_detonate_cd, 2.4)
        self.assertAlmostEqual(quick.truck_max_hp - quick.truck_hp, DRONE_DIRECT_DAMAGE * .85 * .85, places=4)
        self.assertAlmostEqual(base.truck_max_hp - base.truck_hp, DRONE_DIRECT_DAMAGE * .85, places=4)
        self.assertAlmostEqual(precise.truck_max_hp - precise.truck_hp, DRONE_DIRECT_DAMAGE * 1.2 * .85, places=4)
        self.assertAlmostEqual(precise.explosions[-1].radius, DRONE_EXPLOSION_RADIUS * .75)

    def test_signal_filter(self):
        normal = self.sim(drone="wind_gyro")
        filtered = self.sim(drone="signal_filter")
        for s in (normal, filtered):
            s.ew_active_timer = 4
            s.drone_pos[:] = s.truck_pos + np.array([0, 6, 15])
            s._step_drone(.01, DroneInput())
        self.assertGreater(normal.drone_jam_level, .7)
        self.assertAlmostEqual(filtered.drone_jam_level, normal.drone_jam_level * .4, places=5)

    def test_light_frame_and_rapid_respawn(self):
        light = self.sim(drone="light_frame")
        baseline = self.sim(drone="wind_gyro")
        self.assertEqual(light.drone_max_hp, DRONE_BASE_HP * .75)
        for s in (light, baseline):
            s.drone_vel[:] = [0, 0, 100]
            s._step_drone(.01, DroneInput())
        self.assertGreater(float(np.linalg.norm(light.drone_vel)), float(np.linalg.norm(baseline.drone_vel)))
        fast = self.sim(drone="rapid_respawn")
        self.assertEqual(fast.drone_max_battery, DRONE_BATTERY_SECONDS * .8)
        fast._destroy_drone_in_air("test")
        self.assertEqual(fast.drone_respawn_timer, 1.4)
        fast._spawn_fresh_drone()
        fast._detonate_warhead("test", manual=False)
        self.assertEqual(fast.drone_respawn_timer, 1.4)

    def test_heavy_armor_and_traction(self):
        heavy = self.sim(truck="heavy_armor")
        baseline = self.sim(truck="ballistic_computer")
        self.assertEqual(heavy.truck_max_hp, TRUCK_BASE_HP + 50)
        for s in (heavy, baseline):
            s.truck_speed = 100
            s._step_truck(.01, TruckInput())
        self.assertAlmostEqual(heavy.truck_speed, TRUCK_BASE_MAX_SPEED * .85)
        traction = self.sim(truck="traction_control")
        traction.weather_id = baseline.weather_id = "winter_gale"
        traction.truck_speed = baseline.truck_speed = 0
        traction._step_truck(.1, TruckInput(accel=1))
        baseline._step_truck(.1, TruckInput(accel=1))
        self.assertGreater(traction.truck_speed, baseline.truck_speed * 1.4)

    def test_cooling_shotgun_and_pulse(self):
        cool = self.sim(truck="turret_cooling")
        baseline = self.sim(truck="ballistic_computer")
        for s in (cool, baseline):
            s.turret_heat = .5
            s._step_truck(.1, TruckInput())
        self.assertLess(cool.turret_heat, baseline.turret_heat)
        for s in (cool, baseline):
            s.turret_heat = 0
            s._step_truck(.001, TruckInput(fire_turret=True))
        self.assertAlmostEqual(cool.turret_heat, baseline.turret_heat * .75)
        shotgun = self.sim(truck="rapid_shotgun")
        shotgun._step_truck(.001, TruckInput(fire_shotgun=True))
        self.assertAlmostEqual(shotgun.shotgun_cd, SHOTGUN_COOLDOWN * .6)
        self.assertAlmostEqual(shotgun.projectiles[0].damage, SHOTGUN_PELLET_DMG * .9)
        pulse = self.sim(truck="pulse_jammer")
        pulse._step_truck(.001, TruckInput(trigger_ew=True))
        self.assertAlmostEqual(pulse.ew_cd, EW_JAMMER_COOLDOWN * .65)
        self.assertAlmostEqual(pulse.ew_active_timer, EW_JAMMER_DURATION * .8)

    def test_render_only_dealt_options(self):
        sim = WorldSimulation(seed=12)
        renderer = SceneRenderer3D(800, 520, sim.barriers, os.getcwd())
        class FontSpy:
            def __init__(self, font):
                self.font = font
                self.calls = []
            def render(self, text, *args):
                self.calls.append(text)
                return self.font.render(text, *args)
        spy = FontSpy(renderer.font_sm)
        renderer.font_sm = spy
        renderer.draw_phase_overlay(pygame.Surface((800, 520)), sim, "drone")
        labels = [x for x in spy.calls if x.startswith(("[1]", "[2]", "[3]"))]
        self.assertEqual(len(labels), 3)
        for i, label in enumerate(labels):
            self.assertIn(next(m.name_ru for m in DRONE_MODIFIERS if m.id == sim.drone_draft_options[i]), label)


if __name__ == "__main__":
    unittest.main()
