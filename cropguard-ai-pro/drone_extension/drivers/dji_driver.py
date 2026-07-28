"""
DJI driver - STUB, not functional yet.

DJI does not use MAVLink. Real DJI support requires:
  1. A DJI developer account and one of their SDKs, depending on the
     product line:
       - Mobile SDK (consumer drones, e.g. Mavic/Air/Mini) - Android/
         iOS only, runs the flight app on a phone/tablet paired to
         the remote controller. There is no way to control these
         from a plain Python web backend.
       - Payload SDK (enterprise: Matrice series with a companion
         computer) - C/C++ on an onboard companion computer.
       - Cloud API (DJI Dock / enterprise fleets) - REST-ish, this is
         the only DJI path that fits a backend like this one.
  2. DJI's approval of your developer application (enterprise SDKs
     are not self-serve sign-up).
  3. Actual DJI hardware to test against - none of this can be
     verified without it.

This file defines the correct method signatures (matching
DroneDriver) so that once you have SDK access, you fill in each
method's body against DJI's actual API calls - the rest of the app
(routes, UI) never has to change.
"""
from .base import DroneDriver, DroneDriverError


class DJIDriver(DroneDriver):
    def connect(self, connection_string: str, **kwargs) -> dict:
        raise DroneDriverError(
            "DJI support is not implemented. Requires a DJI developer "
            "account + SDK (Mobile SDK for consumer drones, Payload SDK "
            "for Matrice/enterprise, or Cloud API for Dock) and real "
            "hardware to test against. See this file's docstring."
        )

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return False

    def arm(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def disarm(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def takeoff(self, altitude_m: float) -> dict: raise DroneDriverError("DJI driver not implemented")
    def land(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def return_to_home(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def upload_mission(self, waypoints: list) -> dict: raise DroneDriverError("DJI driver not implemented")
    def start_mission(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def pause_mission(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def resume_mission(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def stop_mission(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def camera_stream_url(self) -> str: raise DroneDriverError("DJI driver not implemented")
    def capture_photo(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def start_video(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def stop_video(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def spray_start(self, rate_lpm: float = None) -> dict: raise DroneDriverError("DJI driver not implemented")
    def spray_stop(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def telemetry(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def health(self) -> dict: raise DroneDriverError("DJI driver not implemented")
    def emergency(self, action: str) -> dict: raise DroneDriverError("DJI driver not implemented")
