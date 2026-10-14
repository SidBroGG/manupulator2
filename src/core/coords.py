"""Coordinate systems and spatial transformation utilities.

Frames of reference:
- World / Table Frame (W): Z = 0 is table plane, normal is (0, 0, 1), manipulator base at (0, 0, 0).
- Robot Base Frame (B): Origin at base of manipulator stand link 'root'.
- Forearm Frame (F): Link '2сустав__1__1__body1' between elbow and wrist joints.
- Gripper / TCP Frame (G): Tool Center Point between finger tips.
- Sensor Frames (C, ToF): Camera optical center and ToF emitter sites.
"""

from __future__ import annotations
from typing import Tuple, Union, Optional
import numpy as np


def translation_matrix(xyz: Union[Tuple[float, float, float], np.ndarray, list]) -> np.ndarray:
    """Create 4x4 homogeneous translation matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def rotation_matrix_x(roll: float) -> np.ndarray:
    """Rotation matrix around X-axis by angle in radians."""
    c, s = np.cos(roll), np.sin(roll)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float64)


def rotation_matrix_y(pitch: float) -> np.ndarray:
    """Rotation matrix around Y-axis by angle in radians."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def rotation_matrix_z(yaw: float) -> np.ndarray:
    """Rotation matrix around Z-axis by angle in radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert extrinsic Euler angles (Roll-Pitch-Yaw / Z-Y-X convention) to 3x3 rotation matrix."""
    rz = rotation_matrix_z(yaw)
    ry = rotation_matrix_y(pitch)
    rx = rotation_matrix_x(roll)
    return rz @ ry @ rx


def pose_to_homogeneous(
    position: Union[Tuple[float, float, float], np.ndarray, list],
    rpy_or_matrix: Optional[Union[Tuple[float, float, float], np.ndarray, list]] = None,
) -> np.ndarray:
    """Construct 4x4 homogeneous transformation matrix from position and orientation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    if rpy_or_matrix is not None:
        mat = np.asarray(rpy_or_matrix, dtype=np.float64)
        if mat.shape == (3, 3):
            T[:3, :3] = mat
        elif mat.shape == (4, 4):
            T = mat.copy()
            T[:3, 3] = np.asarray(position, dtype=np.float64)
        elif len(mat) == 3:
            T[:3, :3] = rpy_to_rotation_matrix(float(mat[0]), float(mat[1]), float(mat[2]))
        else:
            raise ValueError(f"Invalid orientation specification with shape {mat.shape}")
    return T


def homogeneous_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose 4x4 transformation matrix into position vector (3,) and rotation matrix (3, 3)."""
    return T[:3, 3].copy(), T[:3, :3].copy()


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Invert rigid body 4x4 transformation matrix [R | t]^-1 = [R^T | -R^T * t]."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_point(T: np.ndarray, point: Union[Tuple[float, float, float], np.ndarray, list]) -> np.ndarray:
    """Transform 3D point using 4x4 transformation matrix."""
    p_hom = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return (T @ p_hom)[:3]


def transform_vector(T: np.ndarray, vector: Union[Tuple[float, float, float], np.ndarray, list]) -> np.ndarray:
    """Transform 3D direction vector using rotation component of 4x4 transformation matrix."""
    R = T[:3, :3]
    v = np.asarray(vector, dtype=np.float64)
    return R @ v


def rotation_matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in radians."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return float(x), float(y), float(z)
