"""
Автоматическая настройка сетевого доступа к локальному серверу игры.

Модуль решает задачу «друг подключился из другой сети» без Radmin VPN
(у пользователя он не работает) и без playit.gg (нужен премиум-тариф).

Стратегии, применяемые автоматически на стороне хоста:

1. **UPnP/IGD** - программно пробрасываем порт на домашнем роутере
   (AddPortMapping) и узнаём внешний IP. Друг подключается напрямую
   по публичному адресу. Никаких сторонних сервисов, пинга не требуется.
2. **bore.pub** - бесплатный TCP-туннель без регистрации и без премиума.
   Реализация протокола bore целиком на Python (никаких внешних
   бинарников), поэтому другу по-прежнему достаточно одного .exe файла.

Обе стратегии запускаются в фоновых потоках и не блокируют интерфейс.
"""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900
SSDP_TARGETS = (
    "urn:schemas-upnp-org:device:InternetGatewayDevice:1",
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)

UPNP_SERVICE_TYPES = (
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANIPConnection:2",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)

BORE_SERVER = "bore.pub"
BORE_CONTROL_PORT = 7835
BORE_LEGACY_CONTROL_PORT = 2323

PUBLIC_IP_SERVICES = (
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://icanhazip.com",
    "https://ipv4.icanhazip.com",
)


# ---------------------------------------------------------------------------
# UPnP / IGD
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 4.0) -> Optional[str]:
    """GET без лишних зависимостей (stdlib). Возвращает тело ответа."""
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "CrimeanBridge1v1/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = resp.read(65536)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def fetch_public_ip(timeout: float = 5.0) -> Optional[str]:
    """Определяет внешний IP через бесплатные публичные сервисы."""
    for url in PUBLIC_IP_SERVICES:
        body = _http_get(url, timeout=timeout)
        if not body:
            continue
        candidate = body.strip().split()[0] if body.strip() else ""
        if _is_public_ipv4(candidate):
            return candidate
    return None


def _is_public_ipv4(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False
    if any(n < 0 or n > 255 for n in nums):
        return False
    if nums[0] == 10:
        return False
    if nums[0] == 127:
        return False
    if nums[0] == 0:
        return False
    if nums[0] == 169 and nums[1] == 254:
        return False
    if nums[0] == 172 and 16 <= nums[1] <= 31:
        return False
    if nums[0] == 192 and nums[1] == 168:
        return False
    if nums[0] == 100 and 64 <= nums[1] <= 127:
        return False
    return True


def _is_private_ipv4(ip: str) -> bool:
    return not _is_public_ipv4(ip) and bool(ip)


def discover_igd(timeout: float = 3.0) -> Optional[str]:
    """Ищет в локальной сети роутер через SSDP. Возвращает control URL."""
    locations: List[str] = []

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(timeout)
        for target in SSDP_TARGETS:
            msg = (
                "M-SEARCH * HTTP/1.1\r\n"
                f"HOST: {SSDP_ADDR}:{SSDP_PORT}\r\n"
                'MAN: "ssdp:discover"\r\n'
                "MX: 2\r\n"
                f"ST: {target}\r\n"
                "\r\n"
            )
            try:
                sock.sendto(msg.encode("utf-8"), (SSDP_ADDR, SSDP_PORT))
            except OSError:
                pass

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, _ = sock.recvfrom(8192)
            except socket.timeout:
                break
            except OSError:
                break
            text = data.decode("utf-8", errors="replace")
            for line in text.splitlines():
                if line.lower().startswith("location:"):
                    loc = line.split(":", 1)[1].strip()
                    if loc and loc not in locations:
                        locations.append(loc)
                    break
    except OSError:
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass

    for loc in locations:
        control = _parse_igd_description(loc)
        if control:
            return control
    return None


def _parse_igd_description(location: str) -> Optional[str]:
    """Скачивает описание устройства и достаёт controlURL для WAN-сервиса."""
    body = _http_get(location, timeout=4.0)
    if not body:
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None

    base = location.rsplit("/", 1)[0]

    for service in root.iter():
        tag = service.tag.split("}")[-1]
        if tag != "service":
            continue
        service_type = ""
        control_url = ""
        for child in service:
            ctag = child.tag.split("}")[-1]
            if ctag == "serviceType":
                service_type = (child.text or "").strip()
            elif ctag == "controlURL":
                control_url = (child.text or "").strip()
        if not control_url:
            continue
        if any(service_type.startswith(st) for st in UPNP_SERVICE_TYPES):
            return _join_url(base, control_url)
    return None


def _join_url(base: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return base.rstrip("/") + path


def _soap_call(control_url: str, service_type: str, action: str, body_xml: str,
               timeout: float = 5.0) -> Tuple[int, str]:
    """Выполняет SOAP-запрос к роутеру. Возвращает (код, тело ответа)."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(control_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        envelope = (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
            's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
            "<s:Body>" + body_xml + "</s:Body></s:Envelope>"
        ).encode("utf-8")

        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request(
                "POST",
                path,
                body=envelope,
                headers={
                    "Content-Type": 'text/xml; charset="utf-8"',
                    "SOAPAction": f'"{service_type}#{action}"',
                    "Connection": "close",
                },
            )
            resp = conn.getresponse()
            data = resp.read(65536)
            return resp.status, data.decode("utf-8", errors="replace")
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - UPnP может вести себя по-разному
        return 0, str(exc)


def upnp_add_port_mapping(control_url: str, internal_ip: str, port: int,
                          description: str = "CrimeanBridge1v1") -> Tuple[bool, str]:
    """Пробрасывает TCP-порт на роутере. Возвращает (успех, текст ошибки)."""
    service_type = "urn:schemas-upnp-org:service:WANIPConnection:1"
    if "WANPPPConnection" in control_url:
        service_type = "urn:schemas-upnp-org:service:WANPPPConnection:1"
    else:
        # Контрольный URL может принадлежать WANIPConnection:2
        if ":2" in control_url:
            service_type = "urn:schemas-upnp-org:service:WANIPConnection:2"

    body = (
        f'<u:AddPortMapping xmlns:u="{service_type}">'
        "<NewRemoteHost></NewRemoteHost>"
        f"<NewExternalPort>{port}</NewExternalPort>"
        "<NewProtocol>TCP</NewProtocol>"
        f"<NewInternalPort>{port}</NewInternalPort>"
        f"<NewInternalClient>{internal_ip}</NewInternalClient>"
        "<NewEnabled>1</NewEnabled>"
        f"<NewPortMappingDescription>{description}</NewPortMappingDescription>"
        "<NewLeaseDuration>0</NewLeaseDuration>"
        "</u:AddPortMapping>"
    )
    status, text = _soap_call(control_url, service_type, "AddPortMapping", body)
    if status == 200:
        return True, ""
    return False, f"UPnP AddPortMapping: HTTP {status} {text.strip()[:200]}"


def upnp_external_ip(control_url: str) -> Optional[str]:
    """Запрашивает у роутера внешний IP (WAN-адрес)."""
    service_type = "urn:schemas-upnp-org:service:WANIPConnection:1"
    if "WANPPPConnection" in control_url:
        service_type = "urn:schemas-upnp-org:service:WANPPPConnection:1"
    body = f'<u:GetExternalIPAddress xmlns:u="{service_type}"></u:GetExternalIPAddress>'
    status, text = _soap_call(control_url, service_type, "GetExternalIPAddress", body)
    if status != 200:
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    for node in root.iter():
        if node.tag.split("}")[-1] == "NewExternalIPAddress":
            ip = (node.text or "").strip()
            if ip:
                return ip
    return None


def open_port_automatically(port: int, internal_ip: str = "") -> Dict[str, object]:
    """
    Полностью автоматическая попытка открыть порт через UPnP.

    Возвращает словарь:
        ok            - удалось ли пробросить порт
        control_url   - найденный control URL (для повторного вызова)
        external_ip   - внешний IP по данным роутера
        router_ip     - реальный IP WAN на роутере (для определения CGNAT)
        public_ip     - внешний IP по независимому сервису
        cgnat         - True, если провайдер использует CGNAT
        error         - текст ошибки
    """
    result: Dict[str, object] = {
        "ok": False,
        "control_url": "",
        "external_ip": "",
        "router_ip": "",
        "public_ip": "",
        "cgnat": False,
        "error": "",
    }

    control = discover_igd()
    if not control:
        result["error"] = "Роутер не найден (UPnP выключен или есть фаервол)."
        return result
    result["control_url"] = control

    if not internal_ip:
        internal_ip = _primary_lan_ip()

    ok, err = upnp_add_port_mapping(control, internal_ip, port)
    if not ok:
        result["error"] = err
    result["ok"] = ok

    router_ip = upnp_external_ip(control) or ""
    result["router_ip"] = router_ip
    result["external_ip"] = router_ip

    public_ip = fetch_public_ip()
    result["public_ip"] = public_ip or ""

    # CGNAT: адрес, который видит провайдер, не совпадает с WAN-адресом роутера
    if router_ip and public_ip and router_ip != public_ip:
        result["cgnat"] = True
    elif router_ip and _is_private_ipv4(router_ip):
        result["cgnat"] = True

    return result


def _primary_lan_ip() -> str:
    try:
        from .network import get_all_local_ips

        ips = get_all_local_ips()
        for ip in ips:
            if ip.startswith("192.168.") or ip.startswith("10."):
                return ip
        if ips:
            return ips[0]
    except Exception:
        pass
    return "127.0.0.1"


# ---------------------------------------------------------------------------
# bore.pub - бесплатный TCP-туннель без регистрации
# ---------------------------------------------------------------------------


def _recv_bore_message(sock: socket.socket, timeout: float = 6.0) -> Optional[dict]:
    """Читает одно JSON-сообщение bore (delimiter - нулевой байт)."""
    sock.settimeout(timeout)
    buf = bytearray()
    try:
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                return None
            buf.extend(chunk)
            if b"\x00" in buf:
                break
            if len(buf) > 65536:
                return None
    except socket.timeout:
        return None
    except OSError:
        return None
    raw = bytes(buf).split(b"\x00", 1)[0]
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _send_bore_message(sock: socket.socket, payload: dict) -> bool:
    try:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\x00"
        sock.sendall(data)
        return True
    except OSError:
        return False


class BoreTunnel:
    """
    Клиент бесплатного TCP-туннеля bore.pub (без регистрации и премиума).

    Реализует протокол bore целиком на Python, поэтому другу не нужно
    устанавливать никаких дополнительных программ - достаточно .exe игры.
    """

    def __init__(self, server: str = BORE_SERVER, control_port: int = BORE_CONTROL_PORT,
                 local_host: str = "127.0.0.1") -> None:
        self.server = server
        self.control_port = control_port
        self.local_host = local_host
        self.local_port = 0
        self.public_address = ""
        self.error = ""
        self.state = "idle"

        self._control: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- публичный API -----------------------------------------------------
    def start(self, local_port: int, remote_port: int = 0, timeout: float = 8.0) -> bool:
        """Открывает туннель. True - туннель работает."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return bool(self.public_address)
            self.local_port = local_port
            self._stop.clear()
            self.error = ""
            self.state = "connecting"

        ok = self._handshake(timeout=timeout, remote_port=remote_port)
        if not ok:
            # Запасной вариант: старый текстовый протокол bore
            ok = self._handshake_legacy(timeout=timeout, remote_port=remote_port)
        if not ok:
            self.state = "failed"
            return False

        self.state = "connected"
        self._thread = threading.Thread(target=self._control_loop, daemon=True, name="BoreControl")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        ctrl = self._control
        if ctrl is not None:
            try:
                ctrl.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                ctrl.close()
            except OSError:
                pass
        self._control = None
        self.state = "stopped"
        self.public_address = ""

    @property
    def running(self) -> bool:
        return self.state == "connected" and bool(self.public_address)

    # -- внутреннее --------------------------------------------------------
    def _handshake(self, timeout: float, remote_port: int) -> bool:
        try:
            sock = socket.create_connection((self.server, self.control_port), timeout=timeout)
        except OSError as exc:
            self.error = f"Не удалось соединиться с {self.server}:{self.control_port} ({exc})"
            return False

        try:
            # Публичный bore-сервер без пароля сразу ждёт запрос порта.
            if not _send_bore_message(sock, {"Hello": remote_port}):
                sock.close()
                self.error = "Не удалось отправить запрос туннеля."
                return False
            reply = _recv_bore_message(sock, timeout=timeout)
            if reply is None:
                sock.close()
                self.error = "Сервер туннеля не ответил."
                return False
            if "Challenge" in reply:
                sock.close()
                self.error = "Сервер туннеля требует пароль."
                return False
            if "Error" in reply:
                sock.close()
                self.error = str(reply["Error"])
                return False
            port = reply.get("Hello")
            if not isinstance(port, int):
                sock.close()
                self.error = "Неожиданный ответ сервера туннеля."
                return False
            sock.settimeout(None)
            self._control = sock
            self.public_address = f"{self.server}:{port}"
            return True
        except OSError as exc:
            self.error = f"Ошибка рукопожатия с туннелем ({exc})"
            try:
                sock.close()
            except OSError:
                pass
            return False

    def _handshake_legacy(self, timeout: float, remote_port: int) -> bool:
        """Старый текстовый протокол bore (на случай старой версии сервера)."""
        port = BORE_LEGACY_CONTROL_PORT if self.control_port == BORE_CONTROL_PORT else self.control_port
        try:
            sock = socket.create_connection((self.server, port), timeout=timeout)
        except OSError as exc:
            self.error = f"Не удалось соединиться с {self.server}:{port} ({exc})"
            return False
        try:
            sock.sendall(f"L{self.local_port}\n".encode("utf-8"))
            sock.settimeout(timeout)
            data = b""
            while b"\n" not in data and len(data) < 256:
                chunk = sock.recv(64)
                if not chunk:
                    break
                data += chunk
            text = data.decode("utf-8", errors="replace").strip()
            if not text.startswith("OK"):
                sock.close()
                self.error = f"Сервер туннеля ответил: {text or 'ничего'}"
                return False
            public_port = int(text[2:].strip())
            sock.settimeout(None)
            self._control = sock
            self.public_address = f"{self.server}:{public_port}"
            # В старом протоколе данные идут напрямую на выданный порт.
            self._legacy = True
            return True
        except (OSError, ValueError) as exc:
            self.error = f"Ошибка legacy-рукопожатия ({exc})"
            try:
                sock.close()
            except OSError:
                pass
            return False

    def _control_loop(self) -> None:
        ctrl = self._control
        if ctrl is None:
            return
        legacy = getattr(self, "_legacy", False)
        while not self._stop.is_set():
            if legacy:
                # В legacy-режиме клиент сам слушает входящие подключения
                self._accept_legacy()
                continue
            try:
                msg = _recv_bore_message(ctrl, timeout=1.0)
            except OSError:
                break
            if msg is None:
                continue
            if "Connection" in msg:
                conn_id = str(msg["Connection"])
                threading.Thread(
                    target=self._serve_connection, args=(conn_id,), daemon=True,
                    name="BoreConn",
                ).start()
            elif "Error" in msg:
                self.error = str(msg["Error"])
            elif "Hello" in msg:
                continue
        self.state = "stopped"

    def _accept_legacy(self) -> None:
        ctrl = self._control
        if ctrl is None:
            return
        try:
            ctrl.settimeout(1.0)
            conn, _ = ctrl.accept()
        except (socket.timeout, OSError):
            return
        threading.Thread(target=self._pipe, args=(conn, None), daemon=True,
                         name="BoreLegacyConn").start()

    def _serve_connection(self, conn_id: str) -> None:
        """Принимает входящее соединение от bore-сервера и проксирует в игру."""
        # Сначала подключаемся к локальному серверу игры: если он не отвечает,
        # незачем тратить ресурс на стороне туннеля.
        try:
            local = socket.create_connection((self.local_host, self.local_port), timeout=8.0)
        except OSError as exc:
            self.error = f"Локальный сервер игры недоступен ({exc})"
            return

        try:
            data_sock = socket.create_connection((self.server, self.control_port), timeout=8.0)
        except OSError as exc:
            self.error = f"Не удалось принять туннель ({exc})"
            try:
                local.close()
            except OSError:
                pass
            return

        try:
            if not _send_bore_message(data_sock, {"Accept": conn_id}):
                raise OSError("не удалось отправить Accept")
        except OSError as exc:
            self.error = f"Ошибка приёма туннеля ({exc})"
            for s in (data_sock, local):
                try:
                    s.close()
                except OSError:
                    pass
            return

        # Данные, которые сервер успел прислать вместе с Accept.
        try:
            data_sock.settimeout(0.3)
            chunk = data_sock.recv(65536)
            if chunk:
                local.sendall(chunk)
        except OSError:
            pass

        # Возвращаем блокирующий режим: иначе перекачка оборвётся при простое.
        data_sock.settimeout(None)
        local.settimeout(None)

        self._pipe(data_sock, local)

    def _pipe(self, a: socket.socket, b: Optional[socket.socket]) -> None:
        """Двунаправленная перекачка данных между двумя сокетами."""
        if b is None:
            # legacy: клиент -> локальный сервер игры
            try:
                local = socket.create_connection((self.local_host, self.local_port), timeout=8.0)
            except OSError:
                try:
                    a.close()
                except OSError:
                    pass
                return
            b = local

        def _copy(src: socket.socket, dst: socket.socket) -> None:
            try:
                src.settimeout(None)
                dst.settimeout(None)
            except OSError:
                pass
            try:
                while not self._stop.is_set():
                    chunk = src.recv(65536)
                    if not chunk:
                        break
                    dst.sendall(chunk)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        t1 = threading.Thread(target=_copy, args=(a, b), daemon=True)
        t2 = threading.Thread(target=_copy, args=(b, a), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Оркестратор: автоматический подбор доступного способа подключения
# ---------------------------------------------------------------------------


class AutoHostSetup:
    """
    Автоматически готовит публичный доступ к локальному серверу.

    Порядок действий:
      1) UPnP проброс порта + определение внешнего IP (прямое подключение).
      2) Если UPnP не сработал или интернет за CGNAT - бесплатный туннель bore.pub.

    Объект работает в фоновом потоке, GUI не блокируется.
    """

    def __init__(self, port: int, local_ip: str = "",
                 on_update: Optional[Callable[[Dict[str, object]], None]] = None) -> None:
        self.port = port
        self.local_ip = local_ip or _primary_lan_ip()
        self.on_update = on_update
        self.tunnel = BoreTunnel()
        self.addresses: List[Tuple[str, str]] = []   # (подпись, адрес)
        self.status = "Ожидание..."
        self.done = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="AutoHostSetup")
        self._thread.start()

    def _notify(self) -> None:
        if self.on_update is not None:
            try:
                self.on_update(self.snapshot())
            except Exception:
                pass

    def snapshot(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "addresses": list(self.addresses),
            "done": self.done,
        }

    def stop(self) -> None:
        self.tunnel.stop()

    def _run(self) -> None:
        # --- Шаг 1: UPnP ---------------------------------------------------
        self.status = "Пробуем открыть порт через UPnP..."
        self._notify()
        try:
            upnp = open_port_automatically(self.port, self.local_ip)
        except Exception as exc:  # noqa: BLE001
            upnp = {"ok": False, "error": str(exc), "public_ip": "", "cgnat": False}

        direct = ""
        if upnp.get("ok"):
            public_ip = str(upnp.get("public_ip") or upnp.get("external_ip") or "")
            if public_ip and not upnp.get("cgnat"):
                direct = f"{public_ip}:{self.port}"
                self.addresses.append(("Прямое подключение (UPnP)", direct))
                self.status = "Порт открыт на роутере, подключение напрямую."
            else:
                self.status = "Порт открыт, но провайдер использует CGNAT - включаем туннель."
        else:
            self.status = "UPnP не сработал - включаем бесплатный туннель."
        self._notify()

        # --- Шаг 2: бесплатный туннель ------------------------------------
        if not direct:
            self.status = "Поднимаем бесплатный туннель (bore.pub)..."
            self._notify()
            try:
                ok = self.tunnel.start(self.port)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self.tunnel.error = str(exc)
            if ok and self.tunnel.public_address:
                self.addresses.append(("Бесплатный туннель (bore.pub)", self.tunnel.public_address))
                self.status = "Туннель работает."
            else:
                self.status = (
                    "Автоматический доступ не настроен. "
                    "Проверьте интернет или подключитесь к одной Wi-Fi сети."
                )
        self.done = True
        self._notify()
