"""
Тест UPnP/IGD на mock-роутере.

Запуск:  python -m tools.test_upnp
Проверяет: SSDP-поиск, разбор описания устройства, AddPortMapping,
определение внешнего IP и определение CGNAT.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.net_setup as net_setup  # noqa: E402

DEVICE_XML = """<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
  <device>
    <deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>
    <friendlyName>Mock Router</friendlyName>
    <serviceList>
      <service>
        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>
        <serviceId>urn:upnp-org:serviceId:WANIPConn1</serviceId>
        <controlURL>/ctl/IPConn</controlURL>
        <SCPDURL>/scpd.xml</SCPDURL>
      </service>
    </serviceList>
  </device>
</root>
"""

SOAP_RESPONSE = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
<s:Body><u:AddPortMappingResponse xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
</u:AddPortMappingResponse></s:Body></s:Envelope>
"""

EXTERNAL_IP_RESPONSE = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
<s:Body><u:GetExternalIPAddressResponse xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
<NewExternalIPAddress>{ip}</NewExternalIPAddress>
</u:GetExternalIPAddressResponse></s:Body></s:Envelope>
"""


class MockRouter:
    """Роутер: отвечает на SSDP, отдаёт описание устройства и SOAP."""

    def __init__(self, external_ip: str = "203.0.113.77") -> None:
        self.external_ip = external_ip
        self.http_port = 0
        self.ssdp_sock: "socket.socket | None" = None
        self.http_sock: "socket.socket | None" = None
        self.add_port_calls: list = []
        self._stop = threading.Event()

    def start(self) -> None:
        # SSDP
        self.ssdp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.ssdp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ssdp_sock.bind(("0.0.0.0", 0))
        self.ssdp_port = self.ssdp_sock.getsockname()[1]
        threading.Thread(target=self._serve_ssdp, daemon=True).start()

        # HTTP
        self.http_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.http_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.http_sock.bind(("127.0.0.1", 0))
        self.http_sock.listen(8)
        self.http_port = self.http_sock.getsockname()[1]
        threading.Thread(target=self._serve_http, daemon=True).start()

    def _serve_ssdp(self) -> None:
        assert self.ssdp_sock is not None
        while not self._stop.is_set():
            try:
                self.ssdp_sock.settimeout(0.3)
                data, addr = self.ssdp_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if b"M-SEARCH" not in data:
                continue
            resp = (
                "HTTP/1.1 200 OK\r\n"
                "CACHE-CONTROL: max-age=1800\r\n"
                "LOCATION: http://127.0.0.1:%d/rootDesc.xml\r\n"
                "SERVER: MockRouter/1.0 UPnP/1.1\r\n"
                "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n"
                "USN: uuid:mock-router::urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n"
                "\r\n"
            ) % self.http_port
            try:
                self.ssdp_sock.sendto(resp.encode("utf-8"), addr)
            except OSError:
                pass

    def _serve_http(self) -> None:
        assert self.http_sock is not None
        while not self._stop.is_set():
            try:
                self.http_sock.settimeout(0.3)
                conn, _ = self.http_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_http, args=(conn,), daemon=True).start()

    def _handle_http(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(3.0)
            data = b""
            while b"\r\n\r\n" not in data and len(data) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            if b"\r\n\r\n" in data:
                head, _, body = data.partition(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1].strip())
                while len(body) < length:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    body += chunk
                text = head.decode("utf-8", errors="replace")
                body_text = body.decode("utf-8", errors="replace")

                if "rootDesc.xml" in text:
                    payload = DEVICE_XML.encode("utf-8")
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\nConnection: close\r\n\r\n"
                        + payload
                    )
                    return
                if "AddPortMapping" in body_text:
                    self.add_port_calls.append(body_text)
                    payload = SOAP_RESPONSE.encode("utf-8")
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\nConnection: close\r\n\r\n"
                        + payload
                    )
                    return
                if "GetExternalIPAddress" in body_text:
                    payload = EXTERNAL_IP_RESPONSE.format(ip=self.external_ip).encode("utf-8")
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\nConnection: close\r\n\r\n"
                        + payload
                    )
                    return
            conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        for s in (self.ssdp_sock, self.http_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


def run_test() -> int:
    router = MockRouter(external_ip="203.0.113.77")
    router.start()
    print(f"[mock] Роутер: SSDP-порт {router.ssdp_port}, HTTP-порт {router.http_port}")

    # Подменяем адреса SSDP/HTTP на адреса mock-роутера
    orig_discover = net_setup.discover_igd

    def patched_discover(timeout: float = 3.0):
        control = net_setup._parse_igd_description(f"http://127.0.0.1:{router.http_port}/rootDesc.xml")
        return control

    net_setup.discover_igd = patched_discover

    # Внешний IP через сервис - подменяем на заведомо другой (имитация CGNAT)
    orig_fetch = net_setup.fetch_public_ip
    net_setup.fetch_public_ip = lambda timeout=5.0: "198.51.100.23"

    try:
        result = net_setup.open_port_automatically(27015, "192.168.1.50")
    finally:
        net_setup.discover_igd = orig_discover
        net_setup.fetch_public_ip = orig_fetch

    print(f"[1] control_url: {result['control_url']}")
    print(f"[2] ok (порт проброшен): {result['ok']}, ошибка: {result['error']!r}")
    print(f"[3] IP роутера: {result['router_ip']}, публичный IP: {result['public_ip']}")
    print(f"[4] CGNAT определён: {result['cgnat']}")
    print(f"[5] Вызовов AddPortMapping: {len(router.add_port_calls)}")

    problems = []
    if "ctl/IPConn" not in str(result["control_url"]):
        problems.append("не найден controlURL")
    if not result["ok"]:
        problems.append(f"порт не проброшен: {result['error']}")
    if len(router.add_port_calls) != 1:
        problems.append("AddPortMapping вызван не один раз")
    else:
        body = router.add_port_calls[0]
        for expected in ("<NewExternalPort>27015</NewExternalPort>",
                          "<NewProtocol>TCP</NewProtocol>",
                          "<NewInternalPort>27015</NewInternalPort>",
                          "<NewInternalClient>192.168.1.50</NewInternalClient>"):
            if expected not in body:
                problems.append(f"в SOAP-запросе нет {expected}")
    if not result["cgnat"]:
        problems.append("CGNAT не определён (роутер и внешний IP различаются)")
    if result["router_ip"] != "203.0.113.77":
        problems.append("внешний IP роутера прочитан неверно")

    router.stop()
    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print("[OK] UPnP: поиск роутера, проброс порта, внешний IP и определение CGNAT работают.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
