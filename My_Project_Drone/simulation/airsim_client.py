"""
AirSim Python API wrapper — connects to Blocks.exe on port 41451.

Separate from the PX4 physics connection (TCP 4560). This API lets us:
  - Spawn / move actors (victim, attacker spheres)
  - Read camera frames from drone cameras
  - Set weather (fog, rain, wind)
  - Read drone NED position

Gracefully no-ops if the airsim package is not installed or Blocks is not running.
Install: pip install airsim
"""
from __future__ import annotations

import numpy as np

try:
    import airsim as _air
    _HAS_AIRSIM = True
except ImportError:
    _air = None
    _HAS_AIRSIM = False


class AirSimClient:
    """
    Thin wrapper around airsim.MultirotorClient.
    All methods are safe to call even when AirSim is unavailable —
    they return None/False and print a one-time warning.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 41451):
        self._c = None
        self.available = False

        if not _HAS_AIRSIM:
            print("  [AirSim] 'airsim' package not installed — visual actors disabled")
            print("           pip install airsim")
            return

        try:
            self._c = _air.MultirotorClient(ip=ip, port=port)
            self._c.confirmConnection()
            self.available = True
            print(f"  [AirSim] API connected  ({ip}:{port})")
        except Exception as e:
            print(f"  [AirSim] API unavailable ({e}) — visual actors disabled")

    # ── Actor management ──────────────────────────────────────────────────────

    def spawn_sphere(
        self,
        name: str,
        x: float, y: float, z: float = -0.5,
        scale: float = 1.0,
    ) -> bool:
        """
        Spawn a sphere at NED (x=north, y=east, z=down) position.
        z=-0.5 places the sphere 0.5 m above ground.
        Returns True if successful.
        """
        if not self.available:
            return False
        try:
            pose  = _air.Pose(_air.Vector3r(x, y, z),
                              _air.to_quaternion(0, 0, 0))
            scale_vec = _air.Vector3r(scale, scale, scale)
            self._c.simSpawnObject(name, "Sphere", pose, scale_vec,
                                   physics_enabled=False, is_blueprint=False)
            return True
        except Exception as e:
            print(f"  [AirSim] spawn '{name}' failed: {e}")
            return False

    def move_actor(self, name: str, x: float, y: float, z: float = -0.5) -> bool:
        """Teleport an existing actor to NED position."""
        if not self.available:
            return False
        try:
            pose = _air.Pose(_air.Vector3r(x, y, z),
                             _air.to_quaternion(0, 0, 0))
            self._c.simSetObjectPose(name, pose, teleport=True)
            return True
        except Exception:
            return False

    # ── Camera ────────────────────────────────────────────────────────────────

    def get_rgb_frame(
        self,
        vehicle: str = "PX4",
        camera: str = "front_center",
    ) -> "np.ndarray | None":
        """
        Return latest RGB frame from the drone camera as a BGR numpy array
        (OpenCV convention).  Returns None if AirSim unavailable or no cameras
        configured in settings.json.

        compress=True  → AirSim sends PNG bytes → cv2.imdecode works correctly.
        compress=False → raw pixel bytes → cv2.imdecode fails silently (old bug).
        Fallback: if compressed decode fails, try reshape from raw bytes.
        """
        if not self.available:
            return None
        try:
            import cv2

            # Request compressed PNG — imdecode handles this correctly
            resp = self._c.simGetImages([
                _air.ImageRequest(camera, _air.ImageType.Scene, False, True)
            ], vehicle_name=vehicle)

            if not resp or not resp[0].image_data_uint8:
                return None

            arr = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)

            # Compressed path (PNG/JPEG)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame

            # Fallback: raw RGB bytes → reshape → BGR for OpenCV
            h, w = resp[0].height, resp[0].width
            if arr.size == h * w * 3:
                return cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        except Exception:
            pass
        return None

    def get_depth_frame(
        self,
        vehicle: str = "PX4",
        camera: str = "front_center",
    ) -> "np.ndarray | None":
        """Return depth image as float32 array (metres)."""
        if not self.available:
            return None
        try:
            resp = self._c.simGetImages([
                _air.ImageRequest(camera, _air.ImageType.DepthPerspective, True, False)
            ], vehicle_name=vehicle)
            if resp and resp[0].image_data_float:
                return _air.get_pfm_array(resp[0])
        except Exception:
            pass
        return None

    # ── Weather ───────────────────────────────────────────────────────────────

    def set_weather(self, fog: float = 0.0, rain: float = 0.0) -> None:
        """Set fog (0–1) and rain (0–1) in AirSim."""
        if not self.available:
            return
        try:
            self._c.simEnableWeather(True)
            self._c.simSetWeatherParameter(_air.WeatherParameter.Fog,  fog)
            self._c.simSetWeatherParameter(_air.WeatherParameter.Rain, rain)
        except Exception:
            pass

    # ── Drone state ───────────────────────────────────────────────────────────

    def get_drone_ned(self, vehicle: str = "PX4") -> tuple[float, float, float]:
        """Return drone NED position as (north_m, east_m, down_m)."""
        if not self.available:
            return 0.0, 0.0, 0.0
        try:
            st  = self._c.getMultirotorState(vehicle_name=vehicle)
            pos = st.kinematics_estimated.position
            return pos.x_val, pos.y_val, pos.z_val
        except Exception:
            return 0.0, 0.0, 0.0
