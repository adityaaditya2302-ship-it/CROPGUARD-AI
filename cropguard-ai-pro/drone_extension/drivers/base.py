"""
Universal Drone Driver interface.

Every real driver (MAVLink, DJI, ROS, custom) implements this exact
contract, so the rest of the app (routes, UI) never needs to know
which brand of drone it's talking to. This mirrors the plugin
interface you specified: Connect/Disconnect/Arm/Disarm/Takeoff/Land/
RTL/Pause/Resume/UploadMission/StartMission/StopMission/CameraStream/
CapturePhoto/StartVideo/StopVideo/SprayStart/SprayStop/Telemetry/
Health/Emergency.

*** HONESTY NOTE ***
Only MAVLinkDriver (mavlink_driver.py) is a real, working
implementation, and even that has only been written correctly against
the pymavlink API - not run against real hardware or SITL yet (I have
no drone here to test with). DJIDriver, ROSDriver, and CustomDriver
are stubs: they define the correct method signatures but raise
NotImplementedError, because completing them requires SDKs, developer
accounts, and hardware I don't have access to. See each file's
docstring for exactly what's needed to finish it.
"""
from abc import ABC, abstractmethod


class DroneDriverError(Exception):
    """Raised for any driver-level failure (connection lost, command
    rejected, unsupported feature, etc). Routes should catch this and
    return a clean 4xx/5xx JSON error instead of a raw traceback."""
    pass


class DroneDriver(ABC):
    """Abstract base every real drone driver must implement."""

    # ---- connection -----------------------------------------------
    @abstractmethod
    def connect(self, connection_string: str, **kwargs) -> dict:
        """Open a connection. Returns a dict describing the connected
        system (system_id, firmware, etc)."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    # ---- arm / flight -----------------------------------------------
    @abstractmethod
    def arm(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def disarm(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def takeoff(self, altitude_m: float) -> dict:
        raise NotImplementedError

    @abstractmethod
    def land(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def return_to_home(self) -> dict:
        raise NotImplementedError

    # ---- mission -----------------------------------------------------
    @abstractmethod
    def upload_mission(self, waypoints: list) -> dict:
        """waypoints: list of (lat, lon, alt_m) tuples."""
        raise NotImplementedError

    @abstractmethod
    def start_mission(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def pause_mission(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def resume_mission(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def stop_mission(self) -> dict:
        raise NotImplementedError

    # ---- camera --------------------------------------------------------
    @abstractmethod
    def camera_stream_url(self) -> str:
        """Returns an RTSP/HTTP URL the frontend can point a <video>/
        <img> tag at, or raises DroneDriverError if this drone/driver
        doesn't expose one."""
        raise NotImplementedError

    @abstractmethod
    def capture_photo(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def start_video(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def stop_video(self) -> dict:
        raise NotImplementedError

    # ---- spray ------------------------------------------------------
    @abstractmethod
    def spray_start(self, rate_lpm: float = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def spray_stop(self) -> dict:
        raise NotImplementedError

    # ---- telemetry / health / safety --------------------------------
    @abstractmethod
    def telemetry(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def emergency(self, action: str) -> dict:
        """action: 'return_home' | 'emergency_land' | 'stop_spraying' |
        'stop_mission' | 'kill'."""
        raise NotImplementedError
