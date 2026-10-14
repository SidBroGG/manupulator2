"""Kinematics solver using IKPy with full forward and inverse kinematics support."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import yaml
import numpy as np
import scipy.optimize
import ikpy.chain
import ikpy.link

from src.core.coords import (
    pose_to_homogeneous,
    homogeneous_to_pose,
    invert_homogeneous,
    transform_point,
    rpy_to_rotation_matrix,
)
from src.kinematics.limits import JointLimits


class KinematicsError(Exception):
    """Base exception for kinematics errors."""
    pass


class UnreachablePoseError(KinematicsError):
    """Raised when target pose is outside the robot's physical reachability workspace."""
    pass


class SingularityError(KinematicsError):
    """Raised when manipulator configuration is at or dangerously near a singularity."""
    pass


class IKResult:
    """Encapsulates the result of an inverse kinematics computation."""

    def __init__(
        self,
        success: bool,
        positions: np.ndarray,
        position_error_mm: float,
        orientation_error_deg: float,
        manipulability: float,
        message: str = "",
        iterations: int = 0,
    ) -> None:
        self.success = success
        self.positions = np.asarray(positions, dtype=np.float64)
        self.position_error_mm = float(position_error_mm)
        self.orientation_error_deg = float(orientation_error_deg)
        self.manipulability = float(manipulability)
        self.message = message
        self.iterations = iterations

    def __repr__(self) -> str:
        return (
            f"IKResult(success={self.success}, "
            f"pos_err={self.position_error_mm:.3f}mm, "
            f"ori_err={self.orientation_error_deg:.2f}deg, "
            f"manipulability={self.manipulability:.4f}, "
            f"msg='{self.message}')"
        )


class KinematicsSolver:
    """Forward and inverse kinematics solver for the 6-DOF manipulator with IKPy."""

    ARM_JOINT_NAMES = [
        "revolute_1",
        "revolute_2",
        "revolute_3",
        "revolute_4",
        "revolute_5",
    ]

    HOME_POSITION = np.array([0.0, 0.60, 0.60, 0.60, 0.0], dtype=np.float64)

    # Robot base rotor offset from URDF world root
    BASE_ROTOR_OFFSET = np.array([0.0890121, -4.66195e-07, 7.23668e-05], dtype=np.float64)

    # Gripper TCP offset in gripper body frame
    TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.08], dtype=np.float64)

    # Sensor offsets in gripper link frame (from default_config.yaml / scene)
    CAMERA_LOCAL_OFFSET = np.array([0.035, 0.012, 0.061], dtype=np.float64)
    CAMERA_LOCAL_RPY = np.array([0.0, np.pi, 0.0], dtype=np.float64)

    TOF_LOCAL_OFFSET = np.array([0.035, -0.014, 0.054], dtype=np.float64)
    TOF_LOCAL_RPY = np.array([0.0, np.pi, 0.0], dtype=np.float64)

    def __init__(
        self,
        config_path: str = "config/default_config.yaml",
        urdf_path: Optional[str] = None,
    ) -> None:
        self.config = self._load_config(config_path)
        self.limits = JointLimits.from_config(self.config)

        # Build IKPy chains
        self.tcp_chain = self._build_chain(endpoint="gripper_tcp")
        self.forearm_chain = self._build_chain(endpoint="forearm")

        # Table height from config
        self.table_z = float(self.config.get("vision", {}).get("table_plane", {}).get("z_height", 0.0))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _build_chain(self, endpoint: str = "gripper_tcp") -> ikpy.chain.Chain:
        """Construct IKPy kinematic chain with verified URDF parameters."""
        links = [
            ikpy.link.OriginLink(),
            ikpy.link.URDFLink(
                name="rotor_base",
                origin_translation=self.BASE_ROTOR_OFFSET,
                origin_orientation=[0.0, 0.0, 0.0],
                joint_type="fixed",
            ),
            ikpy.link.URDFLink(
                name="revolute_1",
                origin_translation=[0.0, 0.0, 0.014],
                origin_orientation=[np.pi / 2.0, 0.0, 0.0],
                rotation=[0.0, -1.0, 0.0],
                bounds=(-np.pi, np.pi),
                joint_type="revolute",
            ),
            ikpy.link.URDFLink(
                name="revolute_2",
                origin_translation=[0.000223271, 0.0474832, -0.0193178],
                origin_orientation=[-np.pi / 2.0, 0.0, 0.0],
                rotation=[0.0, 1.0, 0.0],
                bounds=(0.0, np.pi),
                joint_type="revolute",
            ),
            ikpy.link.URDFLink(
                name="revolute_3",
                origin_translation=[0.0, 0.0, 0.114],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0.0, 1.0, 0.0],
                bounds=(0.0, np.pi),
                joint_type="revolute",
            ),
        ]

        if endpoint == "forearm":
            active_mask = [False, False, True, True, True]
            return ikpy.chain.Chain(links=links, active_links_mask=active_mask, name="forearm_chain")

        # Full arm to wrist pitch and claw
        links.extend([
            ikpy.link.URDFLink(
                name="revolute_4",
                origin_translation=[0.0, 0.0, 0.114],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0.0, 1.0, 0.0],
                bounds=(0.0, np.pi),
                joint_type="revolute",
            ),
            ikpy.link.URDFLink(
                name="revolute_5",
                origin_translation=[0.00542673, -0.0193187, 0.0638668],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 1.0],
                bounds=(-np.pi, np.pi),
                joint_type="revolute",
            ),
            ikpy.link.URDFLink(
                name="gripper_tcp",
                origin_translation=self.TCP_LOCAL_OFFSET,
                origin_orientation=[0.0, 0.0, 0.0],
                joint_type="fixed",
            ),
        ])
        active_mask = [False, False, True, True, True, True, True, False]
        return ikpy.chain.Chain(links=links, active_links_mask=active_mask, name="tcp_chain")

    def _to_full_chain_q(self, q: Union[List[float], np.ndarray]) -> List[float]:
        """Wrap 5 joint values into full IKPy vector [origin, rotor, q1..q5, tcp]."""
        q_5 = list(q[:5])
        return [0.0, 0.0] + q_5 + [0.0]

    # -------------------------------------------------------------------------
    # Forward Kinematics (FK) and Transformation Tree
    # -------------------------------------------------------------------------

    def compute_fk(
        self,
        joint_positions: Union[List[float], np.ndarray],
        target_frame: str = "gripper_tcp",
    ) -> np.ndarray:
        """Compute 4x4 homogeneous transformation matrix T_base^target for given joint angles."""
        q_5 = np.asarray(joint_positions[:5], dtype=np.float64)

        if target_frame == "forearm":
            full_q = [0.0, 0.0] + list(q_5[:3])
            return self.forearm_chain.forward_kinematics(full_q)

        full_q = self._to_full_chain_q(q_5)
        T_tcp = self.tcp_chain.forward_kinematics(full_q)

        if target_frame == "gripper_tcp":
            return T_tcp

        # Gripper base link (without TCP offset [0, 0, 0.08])
        T_gripper_base = T_tcp.copy()
        T_gripper_base[:3, 3] -= T_tcp[:3, :3] @ self.TCP_LOCAL_OFFSET

        if target_frame == "gripper_base":
            return T_gripper_base

        if target_frame == "camera":
            R_cam = rpy_to_rotation_matrix(*self.CAMERA_LOCAL_RPY)
            T_cam_local = pose_to_homogeneous(self.CAMERA_LOCAL_OFFSET, R_cam)
            return T_gripper_base @ T_cam_local

        if target_frame == "tof":
            R_tof = rpy_to_rotation_matrix(*self.TOF_LOCAL_RPY)
            T_tof_local = pose_to_homogeneous(self.TOF_LOCAL_OFFSET, R_tof)
            return T_gripper_base @ T_tof_local

        raise ValueError(f"Unknown target frame: '{target_frame}'")

    def get_tcp_pose(
        self,
        joint_positions: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get TCP 3D position vector (3,) and 3x3 rotation matrix in base frame."""
        T = self.compute_fk(joint_positions, target_frame="gripper_tcp")
        return T[:3, 3].copy(), T[:3, :3].copy()

    def get_sensor_poses(
        self,
        joint_positions: Union[List[float], np.ndarray],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get positions and orientations of camera, ToF sensor, and forearm link."""
        res = {}
        for name in ["camera", "tof", "forearm", "gripper_tcp"]:
            T = self.compute_fk(joint_positions, target_frame=name)
            res[name] = (T[:3, 3].copy(), T[:3, :3].copy())
        return res

    # -------------------------------------------------------------------------
    # Reachability Workspace Analysis
    # -------------------------------------------------------------------------

    def is_reachable(
        self,
        target_position: Union[Tuple[float, float, float], np.ndarray, list],
        tolerance_m: float = 0.015,
    ) -> bool:
        """Fast geometric check if target point is within the reachable workspace envelope."""
        pos = np.asarray(target_position, dtype=np.float64)
        # Distance from waist rotation axis
        dx = pos[0] - self.BASE_ROTOR_OFFSET[0]
        dy = pos[1] - self.BASE_ROTOR_OFFSET[1]
        r = np.hypot(dx, dy)
        z = pos[2]

        # Workspace limits for this manipulator geometry
        # Max reach: 0.114 + 0.114 + 0.144 ~ 0.372 from shoulder; ~0.45 from origin
        r_min, r_max = 0.12 - tolerance_m, 0.44 + tolerance_m
        z_min, z_max = -0.05 - tolerance_m, 0.42 + tolerance_m

        if not (r_min <= r <= r_max and z_min <= z <= z_max):
            return False

        # Sphere reachability approximation from shoulder joint
        shoulder_center = self.BASE_ROTOR_OFFSET + np.array([0.0, 0.0, 0.061])
        dist_shoulder = np.linalg.norm(pos - shoulder_center)
        if dist_shoulder > 0.42 + tolerance_m:
            return False

        return True

    def compute_reachability_grid(
        self,
        x_bounds: Tuple[float, float] = (0.16, 0.42),
        y_bounds: Tuple[float, float] = (-0.25, 0.25),
        z_level: float = 0.03,
        resolution: float = 0.03,
        orientation_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Compute 2D reachability map on the table plane at specified Z height."""
        xs = np.arange(x_bounds[0], x_bounds[1] + 1e-5, resolution)
        ys = np.arange(y_bounds[0], y_bounds[1] + 1e-5, resolution)
        grid = np.zeros((len(xs), len(ys)), dtype=bool)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                res = self.solve_ik([x, y, z_level], orientation_mode=orientation_mode)
                if res.success and res.position_error_mm < 3.0:
                    grid[i, j] = True

        return {
            "xs": xs,
            "ys": ys,
            "z_level": z_level,
            "grid": grid,
            "reachable_fraction": float(np.mean(grid)),
        }

    # -------------------------------------------------------------------------
    # Jacobian, Manipulability & Singularity Detection
    # -------------------------------------------------------------------------

    def compute_jacobian(
        self,
        joint_positions: Union[List[float], np.ndarray],
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Calculate 6x5 geometric Jacobian matrix J(q) via numerical differentiation."""
        q0 = np.asarray(joint_positions[:5], dtype=np.float64)
        T0 = self.compute_fk(q0)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        J = np.zeros((6, 5), dtype=np.float64)
        for i in range(5):
            q_plus = q0.copy()
            q_plus[i] += eps
            T_plus = self.compute_fk(q_plus)
            p_plus = T_plus[:3, 3]
            R_plus = T_plus[:3, :3]

            # Linear velocity column J_v
            J[:3, i] = (p_plus - p0) / eps

            # Angular velocity column J_w from rotation difference
            dR = R_plus @ R0.T
            w = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1],
            ]) / (2.0 * eps)
            J[3:, i] = w

        return J

    def manipulability_index(self, joint_positions: Union[List[float], np.ndarray]) -> float:
        """Calculate Yoshikawa manipulability index w = sqrt(det(J_pos * J_pos^T))."""
        J = self.compute_jacobian(joint_positions)
        J_pos = J[:3, :]
        det = np.linalg.det(J_pos @ J_pos.T)
        return float(np.sqrt(max(0.0, det)))

    def is_singular(
        self,
        joint_positions: Union[List[float], np.ndarray],
        threshold: float = 1e-4,
    ) -> bool:
        """Determine if arm posture is near a kinematic singularity."""
        mu = self.manipulability_index(joint_positions)
        return mu < threshold

    def condition_number(self, joint_positions: Union[List[float], np.ndarray]) -> float:
        """Compute condition number kappa(J_pos) = s_max / s_min."""
        J = self.compute_jacobian(joint_positions)
        s = np.linalg.svd(J[:3, :], compute_uv=False)
        if s[-1] < 1e-9:
            return float("inf")
        return float(s[0] / s[-1])

    # -------------------------------------------------------------------------
    # Inverse Kinematics (IK) with Multi-Seed and Minimal Displacement
    # -------------------------------------------------------------------------

    def solve_ik(
        self,
        target_position: Union[Tuple[float, float, float], np.ndarray, list],
        target_orientation: Optional[Union[np.ndarray, List[float]]] = None,
        initial_position: Optional[Union[List[float], np.ndarray]] = None,
        orientation_mode: str = "auto",
        pos_tol_mm: float = 2.0,
        max_iter: int = 150,
    ) -> IKResult:
        """Solve inverse kinematics with posture optimization and minimal joint movement.

        Parameters
        ----------
        target_position : (3,) array-like
            Target TCP coordinates (x, y, z) in meters.
        target_orientation : optional (3,) or (3, 3) array-like
            Target orientation or approach vector.
        initial_position : optional (5,) array-like
            Current joint angles in radians to minimize motion from.
        orientation_mode : str
            - "auto": Tries top_down first, then slanted, then free orientation.
            - "top_down": Gripper approaches vertically downward (-Z vector).
            - "slanted": Gripper approaches at a 45 deg angle towards object.
            - "free": Position-only, orientation unconstrained.
            - "custom": Uses target_orientation vector for TCP approach axis.
        pos_tol_mm : float
            Maximum allowable position residual in millimeters (default: 2.0 mm).
        max_iter : int
            Maximum solver iterations.
        """
        if orientation_mode == "auto":
            # Tier 1: Try strict top-down (ideal for grasping on table)
            res_td = self.solve_ik(
                target_position=target_position,
                target_orientation=target_orientation,
                initial_position=initial_position,
                orientation_mode="top_down",
                pos_tol_mm=pos_tol_mm,
                max_iter=max_iter,
            )
            if res_td.success:
                return res_td

            # Tier 2: Try slanted approach (for flat/elevated objects)
            res_sl = self.solve_ik(
                target_position=target_position,
                target_orientation=target_orientation,
                initial_position=initial_position,
                orientation_mode="slanted",
                pos_tol_mm=pos_tol_mm,
                max_iter=max_iter,
            )
            if res_sl.success:
                return res_sl

            # Tier 3: Free orientation prioritizing exact position
            return self.solve_ik(
                target_position=target_position,
                target_orientation=target_orientation,
                initial_position=initial_position,
                orientation_mode="free",
                pos_tol_mm=pos_tol_mm,
                max_iter=max_iter,
            )

        target_pos = np.asarray(target_position, dtype=np.float64)

        if not self.is_reachable(target_pos):
            return IKResult(
                success=False,
                positions=self.HOME_POSITION.copy(),
                position_error_mm=float(np.linalg.norm(target_pos - self.compute_fk(self.HOME_POSITION)[:3, 3]) * 1000),
                orientation_error_deg=180.0,
                manipulability=0.0,
                message=f"Target {target_pos} is outside reachability workspace envelope.",
            )

        # Determine target approach vector (along TCP +Z axis)
        target_approach_vec: Optional[np.ndarray] = None
        if orientation_mode == "top_down":
            target_approach_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        elif orientation_mode == "slanted":
            # Slanted approach: radial vector towards table center + downward slope
            dx = target_pos[0] - self.BASE_ROTOR_OFFSET[0]
            dy = target_pos[1] - self.BASE_ROTOR_OFFSET[1]
            r_norm = max(1e-4, np.hypot(dx, dy))
            v_rad = np.array([dx / r_norm, dy / r_norm, 0.0])
            target_approach_vec = (v_rad * 0.7071 - np.array([0.0, 0.0, 0.7071]))
            target_approach_vec /= np.linalg.norm(target_approach_vec)
        elif orientation_mode == "custom" and target_orientation is not None:
            ori_arr = np.asarray(target_orientation, dtype=np.float64)
            if ori_arr.ndim == 1 and len(ori_arr) == 3:
                target_approach_vec = ori_arr / np.linalg.norm(ori_arr)
            elif ori_arr.shape == (3, 3):
                target_approach_vec = ori_arr[:, 2]  # Z-axis of rotation matrix

        # Reference starting pose for displacement minimization
        current_q = (
            np.asarray(initial_position[:5], dtype=np.float64)
            if initial_position is not None
            else self.HOME_POSITION.copy()
        )

        # Generate candidate starting seeds to explore solution space
        seeds = self._generate_seeds(target_pos, current_q)

        best_result: Optional[IKResult] = None
        best_cost = float("inf")

        joint_weights = np.array([2.0, 1.5, 1.2, 1.0, 0.5], dtype=np.float64)

        for seed in seeds:
            res = self._solve_single_seed(
                target_pos=target_pos,
                target_approach=target_approach_vec,
                seed=seed,
                orientation_mode=orientation_mode,
                max_iter=max_iter,
            )

            if not res["valid_limits"]:
                continue

            pos_err_mm = res["pos_err_mm"]
            ori_err_deg = res["ori_err_deg"]
            q_sol = res["q"]

            # Check if this solution meets positional precision requirement
            success = pos_err_mm <= pos_tol_mm
            if orientation_mode in ("top_down", "slanted", "custom"):
                success = success and (ori_err_deg <= 15.0)

            # Movement displacement cost
            displacement_cost = float(np.sum(joint_weights * (q_sol - current_q) ** 2))
            manipulability = self.manipulability_index(q_sol)

            # Total cost: favor solutions meeting tolerance, low displacement, high manipulability
            total_cost = (
                displacement_cost
                + pos_err_mm * 5.0
                + (0.0 if success else 1000.0)
                - 50.0 * manipulability
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_result = IKResult(
                    success=success,
                    positions=q_sol,
                    position_error_mm=pos_err_mm,
                    orientation_error_deg=ori_err_deg,
                    manipulability=manipulability,
                    message="Solution found" if success else f"Position residual {pos_err_mm:.2f}mm > {pos_tol_mm}mm",
                    iterations=res["iterations"],
                )

        if best_result is not None:
            return best_result

        # Fallback if all seeds failed
        return IKResult(
            success=False,
            positions=self.HOME_POSITION.copy(),
            position_error_mm=999.0,
            orientation_error_deg=180.0,
            manipulability=0.0,
            message="No feasible kinematic solution found within joint limits.",
        )

    def _generate_seeds(self, target_pos: np.ndarray, current_q: np.ndarray) -> List[np.ndarray]:
        """Generate diverse candidate seeds (current, home, analytical estimates)."""
        seeds = [current_q.copy(), self.HOME_POSITION.copy()]

        # Analytical waist rotation angle towards target
        dx = target_pos[0] - self.BASE_ROTOR_OFFSET[0]
        dy = target_pos[1] - self.BASE_ROTOR_OFFSET[1]
        waist_angle = float(np.arctan2(dy, dx))

        # Elbow-up heuristic seed
        seeds.append(np.array([waist_angle, 0.40, 1.20, 1.00, 0.0], dtype=np.float64))
        # Elbow-down heuristic seed
        seeds.append(np.array([waist_angle, 0.90, 0.40, 1.20, 0.0], dtype=np.float64))
        # High reach seed
        seeds.append(np.array([waist_angle, 0.20, 0.80, 0.70, 0.0], dtype=np.float64))

        return seeds

    def _solve_single_seed(
        self,
        target_pos: np.ndarray,
        target_approach: Optional[np.ndarray],
        seed: np.ndarray,
        orientation_mode: str,
        max_iter: int,
    ) -> Dict[str, Any]:
        """Run bounded non-linear least squares optimization for a specific initial seed."""
        bounds_lower = self.limits.lower.copy()
        bounds_upper = self.limits.upper.copy()

        def residual(q: np.ndarray) -> np.ndarray:
            full_q = self._to_full_chain_q(q)
            fk = self.tcp_chain.forward_kinematics(full_q)
            pos = fk[:3, 3]

            # High weight on position residual (300x mm)
            pos_err = (pos - target_pos) * 300.0

            if orientation_mode == "free" or target_approach is None:
                return pos_err

            # TCP Z-axis (column 2) represents approach direction
            z_axis = fk[:3, 2]
            ori_err = (z_axis - target_approach) * 5.0
            return np.concatenate([pos_err, ori_err])

        clipped_seed = np.clip(seed, bounds_lower + 1e-3, bounds_upper - 1e-3)

        res = scipy.optimize.least_squares(
            residual,
            clipped_seed,
            bounds=(bounds_lower, bounds_upper),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=max_iter,
        )

        q_sol = res.x
        full_q = self._to_full_chain_q(q_sol)
        fk_final = self.tcp_chain.forward_kinematics(full_q)
        final_pos = fk_final[:3, 3]
        pos_err_mm = float(np.linalg.norm(final_pos - target_pos) * 1000.0)

        ori_err_deg = 0.0
        if target_approach is not None:
            z_axis = fk_final[:3, 2]
            dot = np.clip(float(np.dot(z_axis, target_approach)), -1.0, 1.0)
            ori_err_deg = float(np.degrees(np.arccos(dot)))

        valid_limits = self.limits.is_valid(q_sol)

        return {
            "q": q_sol,
            "pos_err_mm": pos_err_mm,
            "ori_err_deg": ori_err_deg,
            "valid_limits": valid_limits,
            "iterations": int(res.nfev),
        }
