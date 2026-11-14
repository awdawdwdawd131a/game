"""
Тестовый mock-сервер bore + проверка клиента BoreTunnel.

Запуск:  python -m tools.test_bore_tunnel
Проверяет: рукопожатие, выдачу порта, приём соединений и перекачку данных.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.net_setup import BoreTunnel  # noqa: E402


def _send(sock: socket.socket, payload: dict) -> None:
    sock.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\x00")


def _recv(sock: socket.socket):
    buf = bytearray()
    while b"\x00" not in buf:
        # Do not swallow tunneled application bytes that arrive in the same
        # TCP segment as the NUL-terminated Accept handshake.
        chunk = sock.recv(1)
        if not chunk:
            return None
        buf.extend(chunk)
    return json.loads(bytes(buf).split(b"\x00", 1)[0].decode("utf-8"))


class MockBoreServer:
    """
    Упрощённый, но протокольно точный сервер bore.

    - control-порт: клиент шлёт {"Hello":0} -> сервер отвечает {"Hello":port};
      по той же ссылке приходят уведомления {"Connection": id};
      по control-порту же клиент открывает соединение с {"Accept": id}.
    - публичный порт: сюда подключается "друг".
    """

    def __init__(self, assigned_port: int = 0) -> None:
        self.assigned_port = assigned_port
        self.listener: "socket.socket | None" = None
        self.public_listener: "socket.socket | None" = None
        self.port = 0
        self.public_port = 0
        self.connections = 0
        self._control_sock: "socket.socket | None" = None
        self._pending: dict = {}
        self._stop = threading.Event()

    def start(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]

        self.public_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.public_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.public_listener.bind(("127.0.0.1", 0))
        self.public_listener.listen(8)
        self.public_port = self.public_listener.getsockname()[1]

        if not self.assigned_port:
            self.assigned_port = self.public_port

        threading.Thread(target=self._serve_control, daemon=True).start()
        threading.Thread(target=self._serve_public, daemon=True).start()

    def _serve_control(self) -> None:
        assert self.listener is not None
        while not self._stop.is_set():
            try:
                self.listener.settimeout(0.3)
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_control, args=(conn,), daemon=True).start()

    def _handle_control(self, conn: socket.socket) -> None:
        try:
            first = _recv(conn)
            if first is None:
                conn.close()
                return
            if "Hello" in first:
                self._control_sock = conn
                _send(conn, {"Hello": self.assigned_port})
                # Держим соединение открытым, как настоящий bore-сервер.
                while not self._stop.is_set():
                    conn.settimeout(0.5)
                    try:
                        msg = _recv(conn)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if msg is None:
                        break
                    if "Accept" in msg:
                        peer = self._pending.pop(str(msg["Accept"]), None)
                        if peer is not None:
                            threading.Thread(
                                target=self._relay, args=(peer, conn), daemon=True
                            ).start()
                conn.close()
                return
            if "Accept" in first:
                peer = self._pending.pop(str(first["Accept"]), None)
                if peer is not None:
                    self._relay(peer, conn)
                    return
            conn.close()
        except OSError:
            pass

    def _serve_public(self) -> None:
        assert self.public_listener is not None
        while not self._stop.is_set():
            try:
                self.public_listener.settimeout(0.3)
                peer, _ = self.public_listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn_id = str(uuid.uuid4())
            self._pending[conn_id] = peer
            ctrl = self._control_sock
            if ctrl is not None:
                try:
                    _send(ctrl, {"Connection": conn_id})
                except OSError:
                    pass

    def _relay(self, peer: socket.socket, tunnel: socket.socket) -> None:
        self.connections += 1

        def _pump(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    chunk = src.recv(65536)
                    if not chunk:
                        break
                    dst.sendall(chunk)
            except OSError:
                pass

        t1 = threading.Thread(target=_pump, args=(peer, tunnel), daemon=True)
        t2 = threading.Thread(target=_pump, args=(tunnel, peer), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        for s in (peer, tunnel):
            try:
                s.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        for lst in (self.listener, self.public_listener):
            if lst is not None:
                try:
                    lst.close()
                except OSError:
                    pass


def run_test() -> int:
    # Локальный "сервер игры"
    game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    game.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    game.bind(("127.0.0.1", 0))
    game.listen(4)
    game_port = game.getsockname()[1]
    got = bytearray()

    def _game_accept() -> None:
        try:
            conn, _ = game.accept()
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                got.extend(chunk)
                conn.sendall(b"PONG")
        except OSError:
            pass

    threading.Thread(target=_game_accept, daemon=True).start()

    srv = MockBoreServer()
    srv.start()
    print(f"[mock] control-порт={srv.port}, публичный порт={srv.public_port}")

    tunnel = BoreTunnel(server="127.0.0.1", control_port=srv.port, local_host="127.0.0.1")
    ok = tunnel.start(game_port)
    print(f"[1] Туннель поднят: {ok}, адрес={tunnel.public_address!r}, ошибка={tunnel.error!r}")
    if not ok:
        return 1

    time.sleep(0.2)
    peer = socket.create_connection(("127.0.0.1", srv.public_port), timeout=5.0)
    peer.sendall(b"HELLO-FRIEND")
    time.sleep(0.8)
    reply = peer.recv(64)
    print(f"[2] Друг отправил HELLO-FRIEND, ответ: {reply!r}")
    print(f"[3] Игра получила: {bytes(got)!r}")
    print(f"[4] Сервер туннеля принял соединений: {srv.connections}")

    peer.close()
    tunnel.stop()
    srv.stop()
    game.close()

    problems = []
    if reply != b"PONG":
        problems.append("нет ответа от игры через туннель")
    if bytes(got) != b"HELLO-FRIEND":
        problems.append("данные не дошли до игры")
    if srv.connections < 1:
        problems.append("туннель не принял соединение")

    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print("[OK] Туннель bore работает: рукопожатие, приём и перекачка данных.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
