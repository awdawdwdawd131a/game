"""Regression tests for slow sockets, coalesced input and adaptive rendering."""
import json
import os
import socket
import threading
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from src.network import _ClientSender, _ServerPeer, GameClient, GameServer, HEADER_STRUCT, encode_packet, recv_packet, send_packet
from src.physics import DroneInput, TruckInput, WorldSimulation
from src.renderer import SceneRenderer3D
from src.net_setup import BoreTunnel
from tools.test_bore_tunnel import MockBoreServer


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.01)
    return False


def decode(frame):
    size = HEADER_STRUCT.unpack(frame[:4])[0]
    return json.loads(frame[4:4 + size])


class BlockingSocket:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.frames = []

    def sendall(self, frame):
        self.entered.set()
        if not self.release.wait(5):
            raise OSError("send timed out")
        self.frames.append(frame)

    def shutdown(self, how):
        self.release.set()

    def close(self):
        self.release.set()


class NetworkPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 520))

    def test_peer_never_queues_stale_states_and_prioritizes_pong(self):
        sock = BlockingSocket()
        peer = _ServerPeer(sock)
        peer.control({"type": "welcome"})
        peer.start()
        self.assertTrue(sock.entered.wait(1))
        for i in range(100):
            peer.state(encode_packet({"type": "state", "n": i}))
        peer.control({"type": "pong", "t": 123})
        sock.release.set()
        self.assertTrue(wait_for(lambda: len(sock.frames) >= 3))
        peer.close()
        packets = list(map(decode, sock.frames))
        self.assertEqual([p["type"] for p in packets], ["welcome", "pong", "state"])
        self.assertEqual(packets[-1]["n"], 99)

    def test_client_coalesces_inputs_without_losing_actions_or_mouse(self):
        for cls, d1, d2, yaw, shot in (
            (DroneInput, {"yaw_delta": .1, "detonate": True}, {"yaw_delta": .2}, "yaw_delta", "detonate"),
            (TruckInput, {"aim_yaw_delta": .1, "fire_shotgun": True}, {"aim_yaw_delta": .2}, "aim_yaw_delta", "fire_shotgun"),
        ):
            with self.subTest(role=cls.__name__):
                sock = BlockingSocket()
                sender = _ClientSender(sock)
                # Keep the worker blocked on the first packet while inputs pile up.
                sender.ping()
                sender.start()
                self.assertTrue(sock.entered.wait(1))
                sender.input(cls(ready=True, pick_modifier="first", **d1).to_dict())
                sender.input(cls(pick_modifier="new", **d2).to_dict())
                sender.input(cls().to_dict())
                self.assertTrue(sender._pending[shot])
                self.assertTrue(sender._pending["ready"])
                self.assertEqual(sender._pending["pick_modifier"], "new")
                self.assertAlmostEqual(sender._pending[yaw], .3)
                sock.release.set()
                self.assertTrue(wait_for(lambda: len(sock.frames) >= 2))
                sent = decode(sock.frames[1])["data"]
                self.assertTrue(sent[shot])
                self.assertTrue(sent["ready"])
                self.assertEqual(sent["pick_modifier"], "new")
                self.assertAlmostEqual(sent[yaw], .3)
                sender.close()

    def test_slow_client_cannot_stop_server_or_other_client(self):
        server = GameServer(host="127.0.0.1", port=0, seed=8)
        slow, fast = GameClient(), GameClient()
        blocked = threading.Event()
        release = threading.Event()
        class SlowSocket:
            def __init__(self, real):
                self.real = real
            def sendall(self, data):
                blocked.set()
                if not release.wait(5):
                    raise OSError("test timeout")
                self.real.sendall(data)
            def shutdown(self, how):
                self.real.shutdown(how)
            def close(self):
                self.real.close()
        try:
            port = server.start()
            self.assertTrue(slow.connect("127.0.0.1", port, "drone"))
            self.assertTrue(fast.connect("127.0.0.1", port, "truck"))
            with server._lock:
                peer = server._peers["drone"]
                peer.sock = SlowSocket(peer.sock)
            self.assertTrue(blocked.wait(2))
            fast.send_truck_input(TruckInput(pick_modifier=server.sim.truck_draft_options[1], ready=True))
            fast.send_ping()
            self.assertTrue(wait_for(lambda: fast.ping_ms > 0))
            self.assertTrue(wait_for(lambda: server.sim.truck_draft_ready))
            latest = []
            def got_state():
                snapshot = fast.pop_latest_snapshot()
                if snapshot:
                    latest.append(snapshot)
                return bool(latest and latest[-1]["truck_draft_ready"])
            self.assertTrue(wait_for(got_state))
            self.assertEqual(latest[-1]["truck_mod"], latest[-1]["truck_draft_options"][1])
            self.assertFalse(release.is_set())
        finally:
            release.set()
            slow.disconnect()
            fast.disconnect()
            server.stop()

    def test_periodic_ping_without_incoming_states(self):
        # The old receiver only scheduled pings after receiving something;
        # this server deliberately sends no state snapshots at all.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        got_ping = threading.Event()
        def serve():
            conn, _ = listener.accept()
            try:
                self.assertEqual(recv_packet(conn)["type"], "hello")
                send_packet(conn, {"type": "welcome", "role": "drone", "server_tick_rate": 60})
                packet = recv_packet(conn)
                if packet and packet.get("type") == "ping":
                    send_packet(conn, {"type": "pong", "t": packet["t"]})
                    got_ping.set()
            finally:
                conn.close()
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        client = GameClient()
        try:
            self.assertTrue(client.connect("127.0.0.1", listener.getsockname()[1], "drone"))
            self.assertTrue(got_ping.wait(3))
            self.assertTrue(wait_for(lambda: client.ping_ms > 0))
        finally:
            client.disconnect()
            listener.close()
            thread.join(1)

    def test_game_through_mock_tunnel_with_two_players(self):
        server = GameServer(host="127.0.0.1", port=0, seed=15)
        mock = MockBoreServer()
        tunnel = BoreTunnel(server="127.0.0.1", local_host="127.0.0.1")
        local, remote = GameClient(), GameClient()
        try:
            port = server.start()
            mock.start()
            tunnel.control_port = mock.port
            self.assertTrue(tunnel.start(port), tunnel.error)
            self.assertTrue(local.connect("127.0.0.1", port, "drone"))
            self.assertTrue(remote.connect("127.0.0.1", mock.public_port, "truck"), remote.error_msg)
            with server._lock:
                d = server.sim.drone_draft_options[1]
                t = server.sim.truck_draft_options[2]
            local.send_drone_input(DroneInput(pick_modifier=d, ready=True))
            remote.send_truck_input(TruckInput(pick_modifier=t, ready=True))
            remote.send_ping()
            self.assertTrue(wait_for(lambda: remote.ping_ms > 0))
            self.assertTrue(wait_for(lambda: server.sim.phase == "ACTIVE_ROUND"))
            self.assertEqual(server.sim.drone_mod, d)
            self.assertEqual(server.sim.truck_mod, t)
            self.assertTrue(local.connected and remote.connected)
        finally:
            local.disconnect()
            remote.disconnect()
            tunnel.stop()
            mock.stop()
            server.stop()

    def test_tunnel_disables_nagle_on_both_relay_sockets(self):
        class RelaySocket:
            def __init__(self):
                self.options = []
            def setsockopt(self, *args):
                self.options.append(args)
            def settimeout(self, timeout):
                pass
            def recv(self, n):
                return b""
            def shutdown(self, how):
                pass
            def close(self):
                pass
        tunnel = BoreTunnel()
        a, b = RelaySocket(), RelaySocket()
        tunnel._pipe(a, b)
        for sock in (a, b):
            self.assertIn((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), sock.options)

    def test_fast_scale_fallback_and_immediate_manual_limit(self):
        sim = WorldSimulation(seed=1)
        renderer = SceneRenderer3D(800, 520, sim.barriers, os.getcwd())
        renderer.set_render_scale(.5)
        self.assertEqual(renderer.render_scale, .5)
        renderer.render_scale = 1.0
        renderer.set_render_scale(1.0)
        for _ in range(8):
            renderer._update_render_scale(120.0)
        self.assertLessEqual(renderer.render_scale, .60001)
        self.assertGreaterEqual(renderer.render_scale, .5)


if __name__ == "__main__":
    unittest.main()
