"""
TCP Multiplayer Server & Client for 2-PC 1v1 Play over LAN / Radmin VPN / ZeroTier / Port-Forwarding.
Uses length-prefixed UTF-8 JSON frames with TCP_NODELAY for low-latency 60 Hz state & input streaming,
plus ping/RTT measurement and role selection (Operator FPV vs Fuel Truck Driver).
"""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Dict, Optional, Tuple

from .bot_ai import BotDroneController, BotTruckController
from .config import DEFAULT_PORT, DT, TICK_RATE
from .physics import DroneInput, TruckInput, WorldSimulation

HEADER_STRUCT = struct.Struct("!I")
MAX_PACKET_BYTES = 256 * 1024


def send_packet(sock: socket.socket, payload: dict) -> bool:
    try:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        sock.sendall(HEADER_STRUCT.pack(len(raw)) + raw)
        return True
    except OSError:
        return False


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_packet(sock: socket.socket) -> Optional[dict]:
    hdr = recv_exact(sock, HEADER_STRUCT.size)
    if hdr is None:
        return None
    (size,) = HEADER_STRUCT.unpack(hdr)
    if size <= 0 or size > MAX_PACKET_BYTES:
        return None
    body = recv_exact(sock, size)
    if body is None:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def get_local_lan_ip() -> str:
    """Best-effort detection of LAN / VPN IPv4 address to display to host player."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def get_all_local_ips() -> list:
    """
    Все IPv4-адреса этого ПК (кроме служебных).

    Нужно, потому что у хоста может быть несколько сетевых интерфейсов:
    обычная домашняя LAN (192.168.x.x), Radmin VPN (26.x.x.x), ZeroTier и т.д.
    Другу надо давать адрес ИМЕННО того интерфейса, через который он
    подключается, поэтому показываем все варианты.
    """
    ips = []
    # 1) Основной интерфейс по маршруту по умолчанию
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            ips.append(ip)
    except OSError:
        pass

    # 2) Все адреса всех интерфейсов (Windows / Linux / macOS)
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["ipconfig"], capture_output=True, text=True, timeout=4, encoding="utf-8", errors="ignore"
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if not line.startswith("IPv4"):
                    continue
                for part in line.replace(":", " ").split():
                    parts = part.split(".")
                    if len(parts) == 4 and all(p.isdigit() for p in parts):
                        addr = ".".join(parts)
                        if not addr.startswith("127.") and addr not in ips:
                            ips.append(addr)
        else:
            out = subprocess.run(
                ["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=4
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if "inet" in parts:
                    idx = parts.index("inet")
                    if idx + 1 < len(parts):
                        addr = parts[idx + 1].split("/")[0]
                        if not addr.startswith("127.") and addr not in ips:
                            ips.append(addr)
    except Exception:
        pass

    return ips


def describe_connection_targets(port: int) -> list:
    """
    Готовые строки «что вписать другу» для подключения к серверу.

    Для каждого сетевого интерфейса показываем адрес и порт. Radmin VPN
    выдаёт адреса вида 26.x.x.x - именно их нужно сообщать другу, если он
    тоже подключён к той же виртуальной сети.
    """
    lines = []
    for ip in get_all_local_ips():
        lines.append(f"{ip}:{port}")
    if not lines:
        lines.append(f"127.0.0.1:{port}")
    return lines


class GameServer:
    """
    Authoritative 1v1 Multiplayer Server.
    Can run either:
    - Embedded in a background thread when Player 1 clicks "Создать сервер (Host)"
    - Standalone headless on any VPS / PC via `python main.py --server --port 27015`
    Supports optional AI bot fill until the second human player connects!
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        fill_with_bot: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.fill_with_bot = fill_with_bot
        self.sim = WorldSimulation(seed=seed)

        self._lock = threading.Lock()
        self._clients: Dict[str, socket.socket] = {}  # role ("drone" | "truck") -> socket
        self._drone_input = DroneInput()
        self._truck_input = TruckInput()
        self._drone_one_shots = {"detonate": False, "ready": False, "pick_modifier": ""}
        self._truck_one_shots = {
            "fire_shotgun": False,
            "trigger_ew": False,
            "trigger_smoke": False,
            "ready": False,
            "pick_modifier": "",
        }
        # Accumulate mouse/stick angular deltas between network packets
        self._acc_drone_yaw = 0.0
        self._acc_drone_pitch = 0.0
        self._acc_truck_yaw = 0.0
        self._acc_truck_pitch = 0.0

        self._bot_drone = BotDroneController(difficulty=0.72)
        self._bot_truck = BotTruckController(difficulty=0.72)

        self._running = False
        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._sim_thread: Optional[threading.Thread] = None
        self.bound_port: int = port

    def start(self) -> int:
        if self._running:
            return self.bound_port
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(4)
        self.bound_port = srv.getsockname()[1]
        self._server_sock = srv
        self._running = True

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="GameServerAccept")
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True, name="GameServerSim")
        self._accept_thread.start()
        self._sim_thread.start()
        return self.bound_port

    def stop(self) -> None:
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        with self._lock:
            for s in list(self._clients.values()):
                try:
                    s.close()
                except OSError:
                    pass
            self._clients.clear()

    def connected_roles(self) -> Tuple[bool, bool]:
        with self._lock:
            return ("drone" in self._clients, "truck" in self._clients)

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while self._running:
            try:
                client_sock, addr = self._server_sock.accept()
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                ).start()
            except OSError:
                break

    def _handle_client(self, sock: socket.socket, addr: Tuple[str, int]) -> None:
        sock.settimeout(10.0)
        hello = recv_packet(sock)
        if not hello or hello.get("type") != "hello":
            sock.close()
            return

        preferred = hello.get("preferred_role", "auto")
        assigned_role: Optional[str] = None

        with self._lock:
            if preferred in ("drone", "truck") and preferred not in self._clients:
                assigned_role = preferred
            elif "drone" not in self._clients:
                assigned_role = "drone"
            elif "truck" not in self._clients:
                assigned_role = "truck"

            if assigned_role is not None:
                self._clients[assigned_role] = sock
                # Reset match cleanly when the second human joins!
                if len(self._clients) == 2:
                    self.sim.reset_full_match()

        if assigned_role is None:
            send_packet(sock, {"type": "reject", "reason": "Сервер полон (оба места 1v1 уже заняты)!"})
            sock.close()
            return

        sock.settimeout(None)
        send_packet(
            sock,
            {
                "type": "welcome",
                "role": assigned_role,
                "server_tick_rate": TICK_RATE,
            },
        )

        try:
            while self._running:
                pkt = recv_packet(sock)
                if pkt is None:
                    break
                ptype = pkt.get("type")
                if ptype == "input":
                    data = pkt.get("data", {})
                    with self._lock:
                        if assigned_role == "drone":
                            din = DroneInput.from_dict(data)
                            self._acc_drone_yaw += din.yaw_delta
                            self._acc_drone_pitch += din.pitch_delta
                            if din.detonate:
                                self._drone_one_shots["detonate"] = True
                            if din.ready:
                                self._drone_one_shots["ready"] = True
                            if din.pick_modifier:
                                self._drone_one_shots["pick_modifier"] = din.pick_modifier
                            self._drone_input = din
                        else:
                            tin = TruckInput.from_dict(data)
                            self._acc_truck_yaw += tin.aim_yaw_delta
                            self._acc_truck_pitch += tin.aim_pitch_delta
                            if tin.fire_shotgun:
                                self._truck_one_shots["fire_shotgun"] = True
                            if tin.trigger_ew:
                                self._truck_one_shots["trigger_ew"] = True
                            if tin.trigger_smoke:
                                self._truck_one_shots["trigger_smoke"] = True
                            if tin.ready:
                                self._truck_one_shots["ready"] = True
                            if tin.pick_modifier:
                                self._truck_one_shots["pick_modifier"] = tin.pick_modifier
                            self._truck_input = tin
                elif ptype == "ping":
                    send_packet(sock, {"type": "pong", "t": pkt.get("t", 0.0)})
        finally:
            with self._lock:
                if self._clients.get(assigned_role) is sock:
                    del self._clients[assigned_role]
            try:
                sock.close()
            except OSError:
                pass

    def _sim_loop(self) -> None:
        next_tick = time.perf_counter()
        tick_idx = 0
        while self._running:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(0.004, next_tick - now))
                continue
            next_tick += DT
            if now - next_tick > 0.25:
                next_tick = now

            with self._lock:
                has_drone = "drone" in self._clients
                has_truck = "truck" in self._clients

                # Wait in draft until both players connect unless bot fill is enabled
                both_ready_to_sim = (has_drone and has_truck) or (self.fill_with_bot and (has_drone or has_truck))

                if has_drone:
                    d_in = DroneInput(
                        throttle=self._drone_input.throttle,
                        forward_Axis=self._drone_input.forward_Axis,
                        strafe_axis=self._drone_input.strafe_axis,
                        yaw_delta=self._acc_drone_yaw,
                        pitch_delta=self._acc_drone_pitch,
                        boost=self._drone_input.boost,
                        detonate=self._drone_one_shots["detonate"],
                        acro_mode=self._drone_input.acro_mode,
                        pick_modifier=self._drone_one_shots["pick_modifier"],
                        ready=self._drone_one_shots["ready"],
                    )
                    self._acc_drone_yaw = 0.0
                    self._acc_drone_pitch = 0.0
                    self._drone_one_shots = {"detonate": False, "ready": False, "pick_modifier": ""}
                elif self.fill_with_bot:
                    d_in = self._bot_drone.compute_input(self.sim, DT)
                else:
                    d_in = DroneInput()

                if has_truck:
                    t_in = TruckInput(
                        accel=self._truck_input.accel,
                        steer=self._truck_input.steer,
                        aim_yaw_delta=self._acc_truck_yaw,
                        aim_pitch_delta=self._acc_truck_pitch,
                        fire_turret=self._truck_input.fire_turret,
                        fire_shotgun=self._truck_one_shots["fire_shotgun"],
                        trigger_ew=self._truck_one_shots["trigger_ew"],
                        trigger_smoke=self._truck_one_shots["trigger_smoke"],
                        pick_modifier=self._truck_one_shots["pick_modifier"],
                        ready=self._truck_one_shots["ready"],
                    )
                    self._acc_truck_yaw = 0.0
                    self._acc_truck_pitch = 0.0
                    self._truck_one_shots = {
                        "fire_shotgun": False,
                        "trigger_ew": False,
                        "trigger_smoke": False,
                        "ready": False,
                        "pick_modifier": "",
                    }
                elif self.fill_with_bot:
                    t_in = self._bot_truck.compute_input(self.sim, DT)
                else:
                    t_in = TruckInput()

                if both_ready_to_sim:
                    self.sim.step(DT, d_in, t_in)
                else:
                    # Hold draft timer while waiting for 2nd PC player to connect
                    if d_in.pick_modifier:
                        self.sim.drone_mod = d_in.pick_modifier
                    if t_in.pick_modifier:
                        self.sim.truck_mod = t_in.pick_modifier
                    self.sim.last_event_ru = "Ожидание подключения 2-го игрока по сети..."

                tick_idx += 1
                # Broadcast snapshot at 30 Hz (every 2 simulation ticks) for crisp smooth play & low bandwidth
                if tick_idx % 2 == 0 and self._clients:
                    snap_pkt = {
                        "type": "state",
                        "players": list(self._clients.keys()),
                        "snap": self.sim.get_snapshot(),
                    }
                    dead = []
                    for r, csock in self._clients.items():
                        if not send_packet(csock, snap_pkt):
                            dead.append(r)
                    for r in dead:
                        try:
                            self._clients[r].close()
                        except OSError:
                            pass
                        del self._clients[r]


class GameClient:
    """Non-blocking threaded TCP client used by desktop players to connect to any Host IP:Port."""

    def __init__(self) -> None:
        self.sock: Optional[socket.socket] = None
        self.role: str = ""
        self.connected: bool = False
        self.error_msg: str = ""
        self.ping_ms: float = 0.0
        self.online_roles: list = []

        self._lock = threading.Lock()
        self._latest_snap: Optional[dict] = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        # Состояние неблокирующего подключения
        self.connecting: bool = False
        self._connect_thread: Optional[threading.Thread] = None
        self._connect_result: Optional[bool] = None
        self._connect_started: float = 0.0
        self._cancel_connect: bool = False

    def connect_async(
        self,
        host: str,
        port: int,
        preferred_role: str = "auto",
        timeout: float = 8.0,
    ) -> bool:
        """
        Запускает подключение В ФОНЕ, не блокируя интерфейс.

        Раньше connect() дёргался напрямую из обработчика событий, и на
        недоступном адресе окно игры висело 5 секунд без всякой реакции -
        у друга это выглядело как «лагает на странице подключения».

        Возвращает True, если подключение реально запущено.
        """
        if self.connecting:
            return False
        self.disconnect()
        self.error_msg = ""
        self._connect_result = None
        self._cancel_connect = False
        self.connecting = True
        self._connect_started = time.perf_counter()

        def _worker() -> None:
            ok = self._connect_blocking(host, port, preferred_role, timeout)
            self._connect_result = ok

        self._connect_thread = threading.Thread(
            target=_worker, daemon=True, name="GameClientConnect"
        )
        self._connect_thread.start()
        return True

    def poll_connect(self) -> Optional[bool]:
        """
        Возвращает результат подключения, когда оно завершилось.

        None - подключение ещё идёт; True - успех; False - ошибка
        (текст ошибки лежит в self.error_msg).
        """
        if not self.connecting:
            return None
        if self._connect_result is None:
            return None
        self.connecting = False
        result = self._connect_result
        self._connect_result = None
        return result

    def connect(self, host: str, port: int, preferred_role: str = "auto", timeout: float = 5.0) -> bool:
        """Синхронное подключение (используется в тестах и headless-сервере)."""
        return self._connect_blocking(host, port, preferred_role, timeout)

    def _connect_blocking(
        self, host: str, port: int, preferred_role: str = "auto", timeout: float = 5.0
    ) -> bool:
        self.disconnect()
        self.error_msg = ""
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_packet(s, {"type": "hello", "preferred_role": preferred_role})
            resp = recv_packet(s)
            if not resp:
                self.error_msg = "Сервер закрыл соединение без ответа."
                s.close()
                return False
            if resp.get("type") == "reject":
                self.error_msg = resp.get("reason", "Отказ сервера")
                s.close()
                return False
            if resp.get("type") != "welcome":
                self.error_msg = "Неверный протокол сервера."
                s.close()
                return False
            if self._cancel_connect:
                # Пользователь нажал ESC, пока шло подключение
                s.close()
                self.error_msg = "Подключение отменено."
                return False

            s.settimeout(None)
            self.sock = s
            self.role = resp["role"]
            self.connected = True
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True, name="GameClientRecv")
            self._recv_thread.start()
            return True
        except socket.timeout:
            self.error_msg = (
                f"Нет ответа от {host}:{port} за {timeout:.0f} сек. "
                "Проверьте IP-адрес и порт, а также что сервер уже запущен."
            )
            return False
        except OSError as exc:
            self.error_msg = f"Не удалось подключиться к {host}:{port} ({exc})"
            return False

    def disconnect(self) -> None:
        self._running = False
        self.connected = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def cancel_connect(self) -> None:
        """Отменяет не завершившееся фоновое подключение."""
        self.connecting = False
        self._connect_result = None
        self._cancel_connect = True

    def send_drone_input(self, inp: DroneInput) -> None:
        if self.connected and self.sock is not None:
            send_packet(self.sock, {"type": "input", "data": inp.to_dict()})

    def send_truck_input(self, inp: TruckInput) -> None:
        if self.connected and self.sock is not None:
            send_packet(self.sock, {"type": "input", "data": inp.to_dict()})

    def send_ping(self) -> None:
        if self.connected and self.sock is not None:
            send_packet(self.sock, {"type": "ping", "t": time.perf_counter()})

    def pop_latest_snapshot(self) -> Optional[dict]:
        with self._lock:
            snap = self._latest_snap
            self._latest_snap = None
            return snap

    def _recv_loop(self) -> None:
        assert self.sock is not None
        last_ping = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            if now - last_ping >= 1.5:
                self.send_ping()
                last_ping = now

            pkt = recv_packet(self.sock)
            if pkt is None:
                self.connected = False
                self.error_msg = "Соединение с сервером потеряно."
                break
            ptype = pkt.get("type")
            if ptype == "state":
                with self._lock:
                    self._latest_snap = pkt.get("snap")
                    self.online_roles = pkt.get("players", [])
            elif ptype == "pong":
                sent_t = float(pkt.get("t", now))
                self.ping_ms = max(0.0, ( time.perf_counter() - sent_t ) * 1000.0)
