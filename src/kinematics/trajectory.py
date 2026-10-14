"""Trajectory planning and interpolation for 6-DOF manipulator joints.

Supports:
- S-curve profile (quintic minimum-jerk polynomial with C2 continuity)
- Trapezoidal profile (velocity-limited with linear acceleration and coasting)
- Multi-joint time synchronization
- Velocity and acceleration limit enforcement
- Waypoint pipeline for pick-and-place task execution
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np

from src.kinematics.limits import JointLimits
from src.kinematics.solver import KinematicsSolver, IKResult, UnreachablePoseError


class TrajectoryPoint:
    """A single discrete point along an interpolated joint trajectory."""

    def __init__(
        self,
        time: float,
        positions: Union[List[float], np.ndarray],
        velocities: Optional[Union[List[float], np.ndarray]] = None,
        accelerations: Optional[Union[List[float], np.ndarray]] = None,
        gripper: float = 0.04,
        segment_name: str = "",
    ) -> None:
        self.time = float(time)
        self.positions = np.asarray(positions[:5], dtype=np.float64)
        self.velocities = (
            np.asarray(velocities[:5], dtype=np.float64)
            if velocities is not None
            else np.zeros(5, dtype=np.float64)
        )
        self.accelerations = (
            np.asarray(accelerations[:5], dtype=np.float64)
            if accelerations is not None
            else np.zeros(5, dtype=np.float64)
        )
        self.gripper = float(gripper)
        self.segment_name = segment_name

    def __repr__(self) -> str:
        return (
            f"TrajectoryPoint(t={self.time:.3f}s, "
            f"pos={np.round(self.positions, 3)}, "
            f"grip={self.gripper:.3f}, seg='{self.segment_name}')"
        )


class Waypoint:
    """A named keyframe in Cartesian or joint space with gripper state and dwell time."""

    def __init__(
        self,
        name: str,
        positions: Union[List[float], np.ndarray],
        gripper: float = 0.04,
        dwell_time: float = 0.0,
        description: str = "",
    ) -> None:
        self.name = name
        self.positions = np.asarray(positions[:5], dtype=np.float64)
        self.gripper = float(gripper)
        self.dwell_time = float(dwell_time)
        self.description = description

    def __repr__(self) -> str:
        return (
            f"Waypoint(name='{self.name}', "
            f"pos={np.round(self.positions, 3)}, "
            f"grip={self.gripper:.3f}, dwell={self.dwell_time}s)"
        )


class TrajectoryGenerator:
    """Generates smooth, velocity- and acceleration-constrained joint trajectories."""

    def __init__(
        self,
        limits: Optional[JointLimits] = None,
        default_profile: str = "s_curve",
        default_dt: float = 0.01,
    ) -> None:
        self.limits = limits if limits is not None else JointLimits()
        self.default_profile = default_profile
        self.default_dt = default_dt

    # -------------------------------------------------------------------------
    # Single Segment Interpolation (Point-to-Point)
    # -------------------------------------------------------------------------

    def plan_segment(
        self,
        q_start: Union[List[float], np.ndarray],
        q_end: Union[List[float], np.ndarray],
        gripper_start: float = 0.04,
        gripper_end: float = 0.04,
        profile: Optional[str] = None,
        dt: Optional[float] = None,
        segment_name: str = "",
        min_duration: float = 0.15,
    ) -> List[TrajectoryPoint]:
        """Interpolate between two joint configurations with time synchronization."""
        p_type = profile if profile is not None else self.default_profile
        step_dt = dt if dt is not None else self.default_dt

        q0 = np.asarray(q_start[:5], dtype=np.float64)
        q1 = np.asarray(q_end[:5], dtype=np.float64)
        dq = q1 - q0

        # Check maximum displacement across all joints
        max_disp = float(np.max(np.abs(dq)))
        if max_disp < 1e-5:
            # Stationary segment
            return [
                TrajectoryPoint(
                    time=0.0,
                    positions=q0,
                    velocities=np.zeros(5),
                    accelerations=np.zeros(5),
                    gripper=gripper_end,
                    segment_name=segment_name,
                )
            ]

        v_max = self.limits.max_velocity
        a_max = self.limits.max_acceleration

        if p_type == "s_curve":
            return self._interpolate_s_curve(
                q0=q0,
                q1=q1,
                v_max=v_max,
                a_max=a_max,
                gripper_start=gripper_start,
                gripper_end=gripper_end,
                dt=step_dt,
                min_duration=min_duration,
                segment_name=segment_name,
            )
        elif p_type == "trapezoidal":
            return self._interpolate_trapezoidal(
                q0=q0,
                q1=q1,
                v_max=v_max,
                a_max=a_max,
                gripper_start=gripper_start,
                gripper_end=gripper_end,
                dt=step_dt,
                min_duration=min_duration,
                segment_name=segment_name,
            )
        else:
            raise ValueError(f"Unknown profile type '{p_type}'. Choose 's_curve' or 'trapezoidal'.")

    def _interpolate_s_curve(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        v_max: np.ndarray,
        a_max: np.ndarray,
        gripper_start: float,
        gripper_end: float,
        dt: float,
        min_duration: float,
        segment_name: str,
    ) -> List[TrajectoryPoint]:
        """Quintic polynomial minimum-jerk profile: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5."""
        dq = q1 - q0

        # Maximum velocity scaling: s_dot_max = 1.875 / T
        T_v = 1.875 * np.abs(dq) / v_max
        # Maximum acceleration scaling: s_ddot_max = 5.7735 / T^2
        T_a = np.sqrt(5.7735 * np.abs(dq) / a_max)

        T_req = float(np.max([np.max(T_v), np.max(T_a)]))
        T = max(T_req, min_duration)

        steps = max(2, int(np.ceil(T / dt)) + 1)
        t_arr = np.linspace(0.0, T, steps)
        tau = t_arr / T

        # Normalized quintic polynomial and derivatives
        s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        s_dot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T
        s_ddot = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (T**2)

        positions = q0[None, :] + s[:, None] * dq[None, :]
        velocities = s_dot[:, None] * dq[None, :]
        accelerations = s_ddot[:, None] * dq[None, :]

        # Gripper interpolation
        grippers = gripper_start + s * (gripper_end - gripper_start)

        points = []
        for i in range(steps):
            points.append(
                TrajectoryPoint(
                    time=float(t_arr[i]),
                    positions=positions[i],
                    velocities=velocities[i],
                    accelerations=accelerations[i],
                    gripper=float(grippers[i]),
                    segment_name=segment_name,
                )
            )
        return points

    def _interpolate_trapezoidal(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        v_max: np.ndarray,
        a_max: np.ndarray,
        gripper_start: float,
        gripper_end: float,
        dt: float,
        min_duration: float,
        segment_name: str,
    ) -> List[TrajectoryPoint]:
        """Synchronized 3-phase trapezoidal profile with linear acceleration and coasting."""
        dq = q1 - q0

        # Required durations for each joint: T_v = |dq| / v_max, T_a = 2 * sqrt(|dq| / a_max)
        T_req = float(np.max(np.maximum(np.abs(dq) / v_max, 2.0 * np.sqrt(np.abs(dq) / a_max))))
        T = max(T_req, min_duration)

        # Acceleration phase duration: 25% of total time or maximum bound
        t_acc = min(0.3 * T, float(np.max(np.abs(dq) / (a_max * T))))
        t_acc = max(0.05 * T, t_acc)
        t_coast = T - 2.0 * t_acc

        steps = max(2, int(np.ceil(T / dt)) + 1)
        t_arr = np.linspace(0.0, T, steps)

        s = np.zeros(steps)
        s_dot = np.zeros(steps)
        s_ddot = np.zeros(steps)

        # Scale factor so that total area under velocity profile is 1.0
        v_cruise = 1.0 / (T - t_acc)
        acc_val = v_cruise / t_acc

        for i, t in enumerate(t_arr):
            if t <= t_acc:
                # Acceleration phase
                s[i] = 0.5 * acc_val * t**2
                s_dot[i] = acc_val * t
                s_ddot[i] = acc_val
            elif t <= t_acc + t_coast:
                # Coasting phase
                t_in_coast = t - t_acc
                s[i] = 0.5 * acc_val * t_acc**2 + v_cruise * t_in_coast
                s_dot[i] = v_cruise
                s_ddot[i] = 0.0
            else:
                # Deceleration phase
                t_in_dec = t - (t_acc + t_coast)
                s[i] = 1.0 - 0.5 * acc_val * (t_acc - t_in_dec)**2
                s_dot[i] = v_cruise - acc_val * t_in_dec
                s_ddot[i] = -acc_val

        s = np.clip(s, 0.0, 1.0)
        positions = q0[None, :] + s[:, None] * dq[None, :]
        velocities = s_dot[:, None] * dq[None, :]
        accelerations = s_ddot[:, None] * dq[None, :]
        grippers = gripper_start + s * (gripper_end - gripper_start)

        points = []
        for i in range(steps):
            points.append(
                TrajectoryPoint(
                    time=float(t_arr[i]),
                    positions=positions[i],
                    velocities=velocities[i],
                    accelerations=accelerations[i],
                    gripper=float(grippers[i]),
                    segment_name=segment_name,
                )
            )
        return points

    # -------------------------------------------------------------------------
    # Waypoint Pipeline Planning (Home -> PreGrasp -> Grasp -> Lift -> Place)
    # -------------------------------------------------------------------------

    def plan_path(
        self,
        waypoints: List[Waypoint],
        profile: Optional[str] = None,
        dt: Optional[float] = None,
    ) -> List[TrajectoryPoint]:
        """Stitch multiple waypoints into a continuous timestamped trajectory."""
        if not waypoints:
            return []

        full_trajectory: List[TrajectoryPoint] = []
        current_time = 0.0

        for i in range(len(waypoints) - 1):
            wp_curr = waypoints[i]
            wp_next = waypoints[i + 1]

            segment = self.plan_segment(
                q_start=wp_curr.positions,
                q_end=wp_next.positions,
                gripper_start=wp_curr.gripper,
                gripper_end=wp_next.gripper,
                profile=profile,
                dt=dt,
                segment_name=f"{wp_curr.name}_to_{wp_next.name}",
            )

            # Offset time
            for p in segment:
                p.time += current_time

            full_trajectory.extend(segment)
            current_time = segment[-1].time

            # Handle dwell pause if requested
            if wp_next.dwell_time > 0:
                dwell_steps = max(1, int(np.ceil(wp_next.dwell_time / (dt or self.default_dt))))
                d_time_span = np.linspace(0.0, wp_next.dwell_time, dwell_steps + 1)[1:]
                for dt_val in d_time_span:
                    full_trajectory.append(
                        TrajectoryPoint(
                            time=current_time + float(dt_val),
                            positions=wp_next.positions,
                            velocities=np.zeros(5),
                            accelerations=np.zeros(5),
                            gripper=wp_next.gripper,
                            segment_name=f"dwell_{wp_next.name}",
                        )
                    )
                current_time += wp_next.dwell_time

        return full_trajectory

    def plan_pick_and_place(
        self,
        pick_position: Union[Tuple[float, float, float], np.ndarray, list],
        dropoff_position: Optional[Union[Tuple[float, float, float], np.ndarray, list]] = None,
        solver: Optional[KinematicsSolver] = None,
        pre_grasp_offset: float = 0.06,
        lift_height: float = 0.12,
        orientation_mode: str = "auto",
        dwell_grasp: float = 0.35,
        dwell_release: float = 0.35,
    ) -> Tuple[List[Waypoint], List[TrajectoryPoint]]:
        """Construct a complete waypoint pipeline and trajectory for pick-and-place.

        Pipeline Stages:
        1. HOME: observation pose, gripper open (0.04m)
        2. PRE_GRASP: positioned directly over object at (Z + pre_grasp_offset), gripper open
        3. GRASP: lowered down to target height, gripper closes with dwell
        4. LIFT: raised vertically up to lift_height, gripper closed
        5. TRANSFER: horizontal movement over drop-off target at lift_height
        6. PLACE: lowered down to drop-off table level, gripper opens with dwell
        7. RETRACT: retracted upward above drop-off
        8. RETURN_HOME: returned back to HOME posture
        """
        ksolver = solver if solver is not None else KinematicsSolver()

        pick_p = np.asarray(pick_position, dtype=np.float64)
        if dropoff_position is not None:
            drop_p = np.asarray(dropoff_position, dtype=np.float64)
        else:
            # Default from config: x: 0.15, y: 0.20, z: 0.05
            drop_cfg = ksolver.config.get("robot", {}).get("dropoff_position", {})
            drop_p = np.array([
                float(drop_cfg.get("x", 0.15)),
                float(drop_cfg.get("y", 0.20)),
                float(drop_cfg.get("z", 0.05)),
            ])

        # Stage coordinates
        p_pre_grasp = np.array([pick_p[0], pick_p[1], pick_p[2] + pre_grasp_offset])
        p_grasp = pick_p.copy()
        p_lift = np.array([pick_p[0], pick_p[1], max(pick_p[2] + lift_height, 0.12)])
        p_transfer = np.array([drop_p[0], drop_p[1], max(drop_p[2] + lift_height, 0.12)])
        p_place = drop_p.copy()
        p_retract = np.array([drop_p[0], drop_p[1], drop_p[2] + pre_grasp_offset])

        # Solve IK for all target stages
        home_q = ksolver.HOME_POSITION.copy()

        ik_pre_grasp = ksolver.solve_ik(p_pre_grasp, initial_position=home_q, orientation_mode=orientation_mode)
        if not ik_pre_grasp.success:
            raise UnreachablePoseError(f"Pre-grasp pose {p_pre_grasp} unreachable: {ik_pre_grasp.message}")

        ik_grasp = ksolver.solve_ik(p_grasp, initial_position=ik_pre_grasp.positions, orientation_mode=orientation_mode)
        if not ik_grasp.success:
            raise UnreachablePoseError(f"Grasp pose {p_grasp} unreachable: {ik_grasp.message}")

        ik_lift = ksolver.solve_ik(p_lift, initial_position=ik_grasp.positions, orientation_mode=orientation_mode)
        if not ik_lift.success:
            raise UnreachablePoseError(f"Lift pose {p_lift} unreachable: {ik_lift.message}")

        ik_transfer = ksolver.solve_ik(p_transfer, initial_position=ik_lift.positions, orientation_mode=orientation_mode)
        if not ik_transfer.success:
            raise UnreachablePoseError(f"Transfer pose {p_transfer} unreachable: {ik_transfer.message}")

        ik_place = ksolver.solve_ik(p_place, initial_position=ik_transfer.positions, orientation_mode=orientation_mode)
        if not ik_place.success:
            raise UnreachablePoseError(f"Place pose {p_place} unreachable: {ik_place.message}")

        ik_retract = ksolver.solve_ik(p_retract, initial_position=ik_place.positions, orientation_mode=orientation_mode)
        if not ik_retract.success:
            raise UnreachablePoseError(f"Retract pose {p_retract} unreachable: {ik_retract.message}")

        # Assemble pipeline waypoints
        waypoints = [
            Waypoint("HOME", home_q, gripper=0.04, description="Initial ready pose"),
            Waypoint("PRE_GRASP", ik_pre_grasp.positions, gripper=0.04, description="Hover above object"),
            Waypoint("GRASP", ik_grasp.positions, gripper=0.00, dwell_time=dwell_grasp, description="Grasp object"),
            Waypoint("LIFT", ik_lift.positions, gripper=0.00, description="Vertical lift clearance"),
            Waypoint("TRANSFER", ik_transfer.positions, gripper=0.00, description="Transfer to drop-off area"),
            Waypoint("PLACE", ik_place.positions, gripper=0.04, dwell_time=dwell_release, description="Release onto table"),
            Waypoint("POST_PLACE", ik_retract.positions, gripper=0.04, description="Retract clearance"),
            Waypoint("RETURN_HOME", home_q, gripper=0.04, description="Return to ready pose"),
        ]

        trajectory = self.plan_path(waypoints)
        return waypoints, trajectory
