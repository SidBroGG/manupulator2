"""Joint limits, safety validation, and boundary enforcement for manipulator joints."""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union
import numpy as np


class JointLimits:
    """Manages physical joint angle bounds, maximum velocities, and safety validation."""

    JOINT_NAMES = [
        "revolute_1",
        "revolute_2",
        "revolute_3",
        "revolute_4",
        "revolute_5",
    ]

    DEFAULT_LOWER = np.array([-np.pi, 0.0, 0.0, 0.0, -np.pi], dtype=np.float64)
    DEFAULT_UPPER = np.array([np.pi, np.pi, np.pi, np.pi, np.pi], dtype=np.float64)
    DEFAULT_MAX_VELOCITY = np.array([2.5, 2.0, 2.0, 2.5, 3.0], dtype=np.float64)  # rad/s
    DEFAULT_MAX_ACCELERATION = np.array([6.0, 5.0, 5.0, 6.0, 8.0], dtype=np.float64)  # rad/s^2

    GRIPPER_LOWER = 0.0   # closed (m)
    GRIPPER_UPPER = 0.04  # open (m)

    def __init__(
        self,
        lower_limits: Optional[Union[List[float], np.ndarray]] = None,
        upper_limits: Optional[Union[List[float], np.ndarray]] = None,
        max_velocities: Optional[Union[List[float], np.ndarray]] = None,
        max_accelerations: Optional[Union[List[float], np.ndarray]] = None,
        gripper_bounds: Tuple[float, float] = (0.0, 0.04),
    ) -> None:
        self.lower = np.asarray(lower_limits if lower_limits is not None else self.DEFAULT_LOWER, dtype=np.float64)
        self.upper = np.asarray(upper_limits if upper_limits is not None else self.DEFAULT_UPPER, dtype=np.float64)
        self.max_velocity = np.asarray(
            max_velocities if max_velocities is not None else self.DEFAULT_MAX_VELOCITY, dtype=np.float64
        )
        self.max_acceleration = np.asarray(
            max_accelerations if max_accelerations is not None else self.DEFAULT_MAX_ACCELERATION, dtype=np.float64
        )
        self.gripper_lower, self.gripper_upper = gripper_bounds

    @classmethod
    def from_config(cls, config: Dict) -> JointLimits:
        """Instantiate JointLimits from parsed configuration dictionary."""
        robot_cfg = config.get("robot", {})
        joints_cfg = robot_cfg.get("joints", [])

        if joints_cfg:
            lower = []
            upper = []
            for j in joints_cfg:
                lower.append(float(j.get("lower_limit", -np.pi)))
                upper.append(float(j.get("upper_limit", np.pi)))
            grip_cfg = robot_cfg.get("gripper", {})
            grip_bounds = (
                float(grip_cfg.get("closed_value", 0.0)),
                float(grip_cfg.get("open_value", 0.04)),
            )
            return cls(lower_limits=lower, upper_limits=upper, gripper_bounds=grip_bounds)
        return cls()

    def is_valid(self, positions: Union[List[float], np.ndarray], tolerance: float = 1e-4) -> bool:
        """Check if all joint positions are within legitimate bounds."""
        pos = np.asarray(positions[:5], dtype=np.float64)
        return bool(np.all(pos >= self.lower - tolerance) and np.all(pos <= self.upper + tolerance))

    def validate(
        self,
        positions: Union[List[float], np.ndarray],
        tolerance: float = 1e-4,
    ) -> Tuple[bool, List[str]]:
        """Validate joint positions and return detailed list of violations."""
        pos = np.asarray(positions[:5], dtype=np.float64)
        violations = []
        for i, name in enumerate(self.JOINT_NAMES):
            if pos[i] < self.lower[i] - tolerance:
                violations.append(
                    f"Joint '{name}' angle {pos[i]:.4f} rad exceeds lower limit {self.lower[i]:.4f} rad"
                )
            elif pos[i] > self.upper[i] + tolerance:
                violations.append(
                    f"Joint '{name}' angle {pos[i]:.4f} rad exceeds upper limit {self.upper[i]:.4f} rad"
                )
        return len(violations) == 0, violations

    def clip(self, positions: Union[List[float], np.ndarray]) -> np.ndarray:
        """Safely clip joint positions to physical limits."""
        pos = np.asarray(positions[:5], dtype=np.float64)
        return np.clip(pos, self.lower, self.upper)

    def is_gripper_valid(self, opening_m: float, tolerance: float = 1e-4) -> bool:
        """Check if gripper opening is within physical range."""
        return (self.gripper_lower - tolerance) <= opening_m <= (self.gripper_upper + tolerance)

    def clip_gripper(self, opening_m: float) -> float:
        """Clip gripper opening to legitimate range."""
        return float(np.clip(opening_m, self.gripper_lower, self.gripper_upper))

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Normalize angle into the range [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def check_velocity_limits(
        self,
        q_start: Union[List[float], np.ndarray],
        q_end: Union[List[float], np.ndarray],
        dt: float,
    ) -> Tuple[bool, np.ndarray]:
        """Check if motion between two configurations exceeds maximum joint velocities."""
        if dt <= 0:
            return False, np.full(5, np.inf)
        q0 = np.asarray(q_start[:5], dtype=np.float64)
        q1 = np.asarray(q_end[:5], dtype=np.float64)
        vel = np.abs(q1 - q0) / dt
        valid = bool(np.all(vel <= self.max_velocity + 1e-4))
        return valid, vel
