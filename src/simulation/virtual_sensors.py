"""Virtual sensor implementations for MuJoCo simulation environment.

Includes:
- VirtualCamera: Emulates the OV5647 camera mounted on the manipulator forearm.
- VirtualToF: Emulates the VL53L0X Time-of-Flight sensor mounted on the forearm.
- PointingGestureInjector: Generates and visualizes 3D human pointing gestures aimed at the table.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import mujoco

from src.core.types import CameraFrame, ToFDistance, Ray3D, PointingGesture


class VirtualCamera:
    """Virtual OV5647 camera attached to the forearm link in MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str = "forearm_camera",
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")

        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture_frame(self) -> CameraFrame:
        """Capture an RGB frame from the forearm camera."""
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb_image = self.renderer.render()
        return CameraFrame(
            width=self.width,
            height=self.height,
            fps=30,
            timestamp=float(self.data.time),
            image=rgb_image,
        )

    def capture_depth(self) -> np.ndarray:
        """Capture depth buffer in meters from forearm camera."""
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return position (x, y, z) and rotation matrix (3x3) of camera in world frame."""
        pos = self.data.cam_xpos[self.cam_id].copy()
        rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return pos, rot

    def close(self) -> None:
        """Clean up renderer resources."""
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()


class VirtualToF:
    """Virtual VL53L0X ToF rangefinder attached to the forearm link in MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str = "tof_sensor_site",
        camera_name: str = "forearm_camera",
        min_range_m: float = 0.03,
        max_range_m: float = 1.20,
        noise_std_m: float = 0.002,
        cone_angle_deg: float = 25.0,
    ) -> None:
        self.model = model
        self.data = data
        self.site_name = site_name
        self.camera_name = camera_name
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.noise_std_m = noise_std_m
        self.cone_angle_rad = np.radians(cone_angle_deg)

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    def _get_ray_directions(self, forward_dir: np.ndarray, right_dir: np.ndarray, up_dir: np.ndarray) -> List[np.ndarray]:
        """Generate multi-ray cone (center + 4 peripheral rays) representing the ToF cone."""
        half_angle = self.cone_angle_rad / 2.0
        tan_half = np.tan(half_angle)

        directions = [forward_dir]  # Center ray

        # Peripheral rays tilted by half-angle
        directions.append((forward_dir + tan_half * right_dir) / np.linalg.norm(forward_dir + tan_half * right_dir))
        directions.append((forward_dir - tan_half * right_dir) / np.linalg.norm(forward_dir - tan_half * right_dir))
        directions.append((forward_dir + tan_half * up_dir) / np.linalg.norm(forward_dir + tan_half * up_dir))
        directions.append((forward_dir - tan_half * up_dir) / np.linalg.norm(forward_dir - tan_half * up_dir))

        return directions

    def measure_distance(self) -> ToFDistance:
        """Raycast into the MuJoCo scene to measure distance to the closest object/surface."""
        origin = self.data.site_xpos[self.site_id].copy()

        # Optical forward direction matches camera optical forward (-Z of camera frame)
        if self.cam_id >= 0:
            cam_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
            right_dir = cam_mat[:, 0]
            up_dir = cam_mat[:, 1]
            forward_dir = -cam_mat[:, 2]
        else:
            site_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
            right_dir = site_mat[:, 0]
            up_dir = site_mat[:, 1]
            forward_dir = -site_mat[:, 2]

        rays = self._get_ray_directions(forward_dir, right_dir, up_dir)
        geom_id = np.zeros(1, dtype=np.int32)

        min_dist = float("inf")
        hit_any = False

        for r_dir in rays:
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                r_dir,
                None,
                1,
                -1,
                geom_id,
            )
            if dist > 0.0 and dist < min_dist:
                min_dist = dist
                hit_any = True

        if not hit_any or min_dist > self.max_range_m:
            return ToFDistance(
                distance_m=self.max_range_m,
                valid=False,
                timestamp=float(self.data.time),
            )

        if min_dist < self.min_range_m:
            return ToFDistance(
                distance_m=self.min_range_m,
                valid=False,
                timestamp=float(self.data.time),
            )

        # Add Gaussian measurement noise (VL53L0X accuracy ~ ±2mm)
        noisy_distance = float(np.clip(min_dist + np.random.normal(0, self.noise_std_m), self.min_range_m, self.max_range_m))

        return ToFDistance(
            distance_m=noisy_distance,
            valid=True,
            timestamp=float(self.data.time),
        )


class PointingGestureInjector:
    """Injects simulated 3D human pointing rays directed at the workspace table."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        table_z: float = 0.0,
        default_origin: Tuple[float, float, float] = (0.55, 0.35, 0.35),
    ) -> None:
        self.model = model
        self.data = data
        self.table_z = table_z
        self.default_origin = default_origin

        self.site_origin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gesture_origin_site")
        self.site_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gesture_target_site")

    def create_gesture(
        self,
        target_xy: Tuple[float, float],
        origin_xyz: Optional[Tuple[float, float, float]] = None,
        confidence: float = 1.0,
    ) -> PointingGesture:
        """Create a 3D ray from the origin targeting a specified table coordinate (x, y)."""
        origin = np.array(origin_xyz if origin_xyz is not None else self.default_origin, dtype=np.float64)
        target = np.array([target_xy[0], target_xy[1], self.table_z], dtype=np.float64)

        delta = target - origin
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            raise ValueError("Origin and target must be distinct points.")

        direction = delta / norm

        # Intersect with table plane Z = table_z
        t = (self.table_z - origin[2]) / direction[2]
        intersection = origin + t * direction

        # Update scene visual sites if present
        if self.site_origin_id >= 0:
            self.model.site_pos[self.site_origin_id] = origin
        if self.site_target_id >= 0:
            self.model.site_pos[self.site_target_id] = [target_xy[0], target_xy[1], self.table_z + 0.002]

        return PointingGesture(
            ray=Ray3D(
                origin=(float(origin[0]), float(origin[1]), float(origin[2])),
                direction=(float(direction[0]), float(direction[1]), float(direction[2])),
            ),
            table_intersection=(float(intersection[0]), float(intersection[1]), float(intersection[2])),
            confidence=confidence,
            timestamp=float(self.data.time),
        )

    def intersect_ray_with_table(self, ray: Ray3D) -> Optional[Tuple[float, float, float]]:
        """Calculate intersection of an arbitrary 3D ray with the table surface plane."""
        dz = ray.direction[2]
        if abs(dz) < 1e-6:
            return None  # Parallel to table plane

        t = (self.table_z - ray.origin[2]) / dz
        if t <= 0:
            return None  # Pointing away from table

        inter_x = ray.origin[0] + t * ray.direction[0]
        inter_y = ray.origin[1] + t * ray.direction[1]
        inter_z = self.table_z
        return (float(inter_x), float(inter_y), float(inter_z))
