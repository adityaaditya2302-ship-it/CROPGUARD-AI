"""
CropGuard AI Pro - Bluetooth Bridge
====================================
Run this alongside app.py (in a second terminal) on a machine with
Bluetooth hardware to let drones/companion devices send images over
classic Bluetooth (RFCOMM / Serial Port Profile) instead of WiFi.

It doesn't reimplement detection - it just receives raw image bytes over
Bluetooth and forwards them to the same /api/drone/upload endpoint the
WiFi path uses, so results, history, and the dashboard all stay in sync
regardless of which connection a given image came in on.

WHY THIS APPROACH:
Bluetooth support varies a lot by OS/hardware, and there's no single
Python library every platform can use for BLE peripheral mode. Classic
Bluetooth RFCOMM (via PyBluez) is the most broadly compatible option for
receiving files from other devices (many companion boards, phones, and
flight controllers support Bluetooth serial). If your drone only speaks
BLE (Bluetooth Low Energy) with a custom GATT profile, use the "Connect
via Bluetooth" button in the web dashboard instead - it uses the Web
Bluetooth API directly in the browser, which handles BLE without needing
any extra software here.

SETUP:
  pip install pybluez2
  python bluetooth_bridge.py

WIRE PROTOCOL (simple by design so any Bluetooth-capable device can
implement it, regardless of platform/language):
  1. Device opens an RFCOMM connection to this machine (SPP, channel 4).
  2. Device sends: 4 bytes (big-endian uint32) = length of image data,
     followed immediately by that many bytes of raw image data (any
     format - JPEG/PNG/HEIC/etc., image_utils.py on the server figures
     out the rest).
  3. This bridge forwards the bytes to /api/drone/upload and, once the
     server responds, sends the JSON result back over the same Bluetooth
     connection so the device can display/log it too (optional for the
     device to read).
"""
import io
import json
import struct
import threading

import requests

SERVER_UPLOAD_URL = 'http://127.0.0.1:5000/api/drone/upload'
RFCOMM_CHANNEL = 4

try:
    import bluetooth  # provided by pybluez2
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False


def _recv_exact(sock, n):
    """Read exactly n bytes from a Bluetooth socket (or None on disconnect)."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def handle_client(client_sock, client_info):
    device_id = f"bt-{client_info[0]}"
    print(f"📡 Bluetooth device connected: {client_info}")
    try:
        while True:
            header = _recv_exact(client_sock, 4)
            if header is None:
                break
            (length,) = struct.unpack('>I', header)
            if length <= 0 or length > 64 * 1024 * 1024:
                print(f"⚠️ Rejecting implausible payload size: {length}")
                break

            image_bytes = _recv_exact(client_sock, length)
            if image_bytes is None:
                break

            print(f"📷 Received {len(image_bytes)} bytes over Bluetooth from {device_id}, forwarding for analysis...")
            try:
                resp = requests.post(
                    SERVER_UPLOAD_URL,
                    files={'image': ('bluetooth_capture.jpg', io.BytesIO(image_bytes))},
                    data={'device_id': device_id, 'connection_type': 'bluetooth'},
                    timeout=30
                )
                result = resp.json()
            except Exception as e:
                result = {'error': str(e)}

            reply = json.dumps(result).encode('utf-8')
            client_sock.sendall(struct.pack('>I', len(reply)) + reply)
            print(f"✅ Sent result back to {device_id}: "
                  f"{result.get('disease', {}).get('name', result.get('error', 'unknown'))}")
    except OSError:
        pass
    finally:
        client_sock.close()
        print(f"📴 Bluetooth device disconnected: {client_info}")


def run_bluetooth_server():
    if not BLUETOOTH_AVAILABLE:
        print("❌ Bluetooth support not installed. Run: pip install pybluez2")
        print("   (Classic Bluetooth also requires a Bluetooth adapter and OS support - "
              "Linux/BlueZ or Windows. On macOS, prefer the browser's Web Bluetooth option instead.)")
        return

    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(('', RFCOMM_CHANNEL))
    server_sock.listen(4)

    bluetooth.advertise_service(
        server_sock, 'CropGuardAI',
        service_id='94f39d29-7d6d-437d-973b-fba39e49d4ee',
        service_classes=[bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE]
    )

    print(f"🔵 Bluetooth bridge listening on RFCOMM channel {RFCOMM_CHANNEL}. "
          f"Pair any Bluetooth-capable drone/companion device with this computer, "
          f"then connect to service 'CropGuardAI' and stream images per the wire protocol above.")

    while True:
        client_sock, client_info = server_sock.accept()
        threading.Thread(target=handle_client, args=(client_sock, client_info), daemon=True).start()


if __name__ == '__main__':
    run_bluetooth_server()
