"""
ROS / ROS2 driver - STUB, not functional yet.

This only makes sense if your drone/companion computer already runs
ROS or ROS2 with a MAVROS (or similar) bridge exposing topics/services
like /mavros/cmd/arming, /mavros/setpoint_position/local, etc. That
means: no ROS-equipped robot, no way to test this - it's environment-
specific by nature (topic names, message types, and QoS settings vary
per robot's setup).

To implement for real you'd typically use `rospy` (ROS1) or `rclpy`
(ROS2) to call MAVROS services and subscribe to its telemetry topics.
Since MAVROS itself sits on top of MAVLink, for many ag-drone setups
it's simpler to just use MAVLinkDriver directly and skip ROS entirely
- reach for this driver only if you already have a broader ROS stack
(e.g. custom obstacle-avoidance nodes, SLAM, multi-robot coordination)
that this app needs to talk to.
"""
from .base import DroneDriver, DroneDriverError


class ROSDriver(DroneDriver):
    def connect(self, connection_string: str, **kwargs) -> dict:
        raise DroneDriverError(
            "ROS driver is not implemented. Requires a running ROS/ROS2 "
            "master with a MAVROS (or equivalent) bridge already up on "
            "your robot, plus rospy/rclpy installed here. See this "
            "file's docstring."
        )

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return False

    def arm(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def disarm(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def takeoff(self, altitude_m: float) -> dict: raise DroneDriverError("ROS driver not implemented")
    def land(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def return_to_home(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def upload_mission(self, waypoints: list) -> dict: raise DroneDriverError("ROS driver not implemented")
    def start_mission(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def pause_mission(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def resume_mission(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def stop_mission(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def camera_stream_url(self) -> str: raise DroneDriverError("ROS driver not implemented")
    def capture_photo(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def start_video(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def stop_video(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def spray_start(self, rate_lpm: float = None) -> dict: raise DroneDriverError("ROS driver not implemented")
    def spray_stop(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def telemetry(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def health(self) -> dict: raise DroneDriverError("ROS driver not implemented")
    def emergency(self, action: str) -> dict: raise DroneDriverError("ROS driver not implemented")
