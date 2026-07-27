"""
Phase 5: DJI adapter - INTERFACE STUB ONLY.

*** HONESTY NOTE ***
DJI agricultural drones (T10/T20/T25/T30/T40/T50) do NOT expose a
public direct-connect Python SDK the way MAVLink does. DJI's official
path is:
  1. DJI Cloud API (for T-series agriculture drones) - requires a
     DJI enterprise developer account, an approved application, and
     a DJI Dock/Pilot 2 app relay - it is a cloud/webhook integration,
     not a direct "plug in over WiFi and call a function" SDK.
  2. DJI Mobile SDK - is for building a *mobile app* that pairs with
     the remote controller; it does not run from a Python backend.

Because of this, there is no honest way to write a working
"connect_dji(device_id)" function without:
  - A DJI enterprise developer account (paid, approval required)
  - Real T-series hardware + DJI Dock/remote controller
  - Following DJI's Cloud API onboarding (OAuth, MQTT topics, webhook
    endpoints) which changes by SDK version

What this file gives you instead: the interface your routes should
call, matching the shape of simulated_telemetry.py / mavlink_adapter.py,
so that once you have DJI enterprise access, you (or your dev team)
fill in the method bodies against DJI's actual Cloud API docs:
https://developer.dji.com/doc/cloud-api-tutorial/en/
"""


class DJINotConfiguredError(Exception):
    pass


def connect(device_id, dji_account_credentials=None):
    raise DJINotConfiguredError(
        "DJI Cloud API is not configured. This requires a DJI enterprise "
        "developer account, an approved app, and a DJI Dock/RC relay. "
        "See https://developer.dji.com/doc/cloud-api-tutorial/en/ "
        "and fill in this adapter once you have credentials."
    )


def disconnect(device_id):
    raise DJINotConfiguredError("DJI Cloud API is not configured.")


def get_telemetry(device_id):
    raise DJINotConfiguredError("DJI Cloud API is not configured.")


def send_mission(device_id, waypoints):
    raise DJINotConfiguredError("DJI Cloud API is not configured.")
