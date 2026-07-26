"""
CropGuard AI Pro - WiFi Auto-Discovery Beacon
==============================================
Most drone companion apps / onboard computers won't know the IP address
of this server ahead of time. Rather than forcing manual IP entry, this
broadcasts a small UDP "here I am" packet on the local WiFi network every
few seconds. Any drone/app that listens on UDP port 41234 for a packet
starting with "CROPGUARD_SERVER:" can auto-discover this server's HTTP
address and start POSTing images to it immediately - works for any drone
brand, since it's just plain UDP + HTTP, not a proprietary SDK.

Protocol (trivial by design, so any device can implement it in a few
lines regardless of platform):
  1. Listen for UDP broadcast packets on port 41234.
  2. Packet payload looks like: "CROPGUARD_SERVER:<ip>:<port>"
  3. POST images (any format) as multipart/form-data field "image" to
     http://<ip>:<port>/api/drone/upload
"""
import socket
import threading
import time

DISCOVERY_PORT = 41234
BROADCAST_INTERVAL_SECONDS = 3


def _get_local_ip():
    """Best-effort local LAN IP (not 127.0.0.1) without needing internet access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


class DiscoveryBeacon:
    """Runs a background thread broadcasting this server's address on the LAN."""

    def __init__(self, http_port=5000):
        self.http_port = http_port
        self._stop_event = threading.Event()
        self._thread = None

    def get_local_ip(self):
        return _get_local_ip()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        while not self._stop_event.is_set():
            try:
                ip = _get_local_ip()
                message = f"CROPGUARD_SERVER:{ip}:{self.http_port}".encode('utf-8')
                sock.sendto(message, ('<broadcast>', DISCOVERY_PORT))
            except OSError:
                pass  # network may be temporarily unavailable; just retry
            time.sleep(BROADCAST_INTERVAL_SECONDS)

        sock.close()


_beacon = None


def start_discovery_beacon(http_port=5000):
    global _beacon
    if _beacon is None:
        _beacon = DiscoveryBeacon(http_port=http_port)
        _beacon.start()
    return _beacon
