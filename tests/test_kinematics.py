"""Tests for kinematics module: FK/IK, transformation trees, reachability, singularities, trajectories."""

import numpy as np
import pytest

from src.core.coords import (
    translation_matrix,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rpy_to_rotation_matrix,
    rotation_matrix_to_rpy,
    pose_to_homogeneous,
    homogeneous_to_pose,
    invert_homogeneous,
    transform_point,
    transform_vector,
)
from src.kinematics.limits import JointLimits
from src.kinematics.solver import (
    KinematicsSolver,
    IKResult,
    KinematicsError,
    UnreachablePoseError,
    SingularityError,
)
from src.kinematics.trajectory import (
    TrajectoryGenerator,
    TrajectoryPoint,
    Waypoint,
)
from src.simulation.mujoco_sim import MujocoSim


class TestCoordinatesAndLimits:
    """Test mathematical coordinate transformations and joint limit validation."""

    def test_coords_transformations(self):
        """Verify translation, rotation, and inversion operations."""
        t = [0.1, -0.2, 0.3]
        T_trans = translation_matrix(t)
        assert np.allclose(T_trans[:3, 3], t)
        assert np.allclose(T_trans[:3, :3], np.eye(3))

        # Inversion check
        T_inv = invert_homogeneous(T_trans)
        assert np.allclose(T_trans @ T_inv, np.eye(4))

        # Euler angle round-trip
        r, p, y = 0.2, -0.4, 0.6
        R = rpy_to_rotation_matrix(r, p, y)
        r_rec, p_rec, y_rec = rotation_matrix_to_rpy(R)
        assert np.isclose(r, r_rec, atol=1e-5)
        assert np.isclose(p, p_rec, atol=1e-5)
        assert np.isclose(y, y_rec, atol=1e-5)

        # Point and vector transform
        T = pose_to_homogeneous([1.0, 2.0, 3.0], [0.0, 0.0, np.pi / 2.0])
        pt = [1.0, 0.0, 0.0]
        transformed_pt = transform_point(T, pt)
        # 90 deg yaw rotates [1, 0, 0] to [0, 1, 0], then translates + [1, 2, 3] -> [1, 3, 3]
        assert np.allclose(transformed_pt, [1.0, 3.0, 3.0], atol=1e-5)

        vec = [1.0, 0.0, 0.0]
        transformed_vec = transform_vector(T, vec)
        # Vector translation is ignored, only rotation applies
        assert np.allclose(transformed_vec, [0.0, 1.0, 0.0], atol=1e-5)

    def test_joint_limits_validation(self):
        """Verify joint limits checking, clipping, and angle wrap-around."""
        limits = JointLimits()

        # Legitimate posture within bounds
        q_valid = [0.0, 0.6, 0.6, 0.6, 0.0]
        assert limits.is_valid(q_valid)
        valid, violations = limits.validate(q_valid)
        assert valid and len(violations) == 0

        # Out-of-bound posture
        q_invalid = [0.0, -0.5, 0.6, 3.5, 0.0]  # revolute_2 < 0, revolute_4 > pi
        assert not limits.is_valid(q_invalid)
        valid, violations = limits.validate(q_invalid)
        assert not valid
        assert len(violations) == 2

        # Safe clipping
        clipped = limits.clip(q_invalid)
        assert limits.is_valid(clipped)
        assert clipped[1] == 0.0  # clipped to lower limit
        assert clipped[3] == np.pi  # clipped to upper limit

        # Gripper validation
        assert limits.is_gripper_valid(0.02)
        assert not limits.is_gripper_valid(0.06)
        assert limits.clip_gripper(0.08) == 0.04
        assert limits.clip_gripper(-0.01) == 0.0

        # Velocity check
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        # dt=0.5 -> 1.0 rad/s (below max_velocity ~2.0 rad/s)
        vel_ok, _ = limits.check_velocity_limits(q0, q1, dt=0.5)
        assert vel_ok
        # dt=0.05 -> 10.0 rad/s (exceeds max_velocity)
        vel_fast, _ = limits.check_velocity_limits(q0, q1, dt=0.05)
        assert not vel_fast


class TestForwardKinematicsAndSensorTree:
    """Test Forward Kinematics alignment with MuJoCo digital twin and sensor frames."""

    @pytest.fixture
    def solver(self):
        return KinematicsSolver()

    @pytest.fixture
    def sim(self):
        s = MujocoSim()
        yield s
        s.close()

    def test_fk_exact_mujoco_alignment(self, solver, sim):
        """Compare IKPy Forward Kinematics against MuJoCo forward kinematics across diverse poses."""
        test_postures = [
            [0.0, 0.60, 0.60, 0.60, 0.0],       # Home pose
            [0.0, 0.0, 0.0, 0.0, 0.0],          # Zero pose
            [0.4, 0.8, 0.5, 0.9, -0.3],         # Asymmetric pose 1
            [-0.5, 0.3, 1.2, 0.4, 0.7],         # Asymmetric pose 2
            [0.8, 1.1, 0.2, 1.4, -0.6],         # Asymmetric pose 3
        ]

        for q in test_postures:
            sim.set_joint_positions(q)
            mujoco_tcp = sim.data.site_xpos[sim.tcp_site_id].copy()
            mujoco_rot = sim.data.site_xmat[sim.tcp_site_id].reshape(3, 3).copy()

            ikpy_pos, ikpy_rot = solver.get_tcp_pose(q)

            # Positional difference must be sub-millimeter (< 0.001 mm)
            pos_diff_m = np.linalg.norm(mujoco_tcp - ikpy_pos)
            assert pos_diff_m < 1e-5, f"Pos diff {pos_diff_m*1000} mm exceeds tolerance at q={q}"

            # Rotational matrix Frobenius norm difference
            rot_diff = np.linalg.norm(mujoco_rot - ikpy_rot)
            assert rot_diff < 1e-4, f"Rot diff {rot_diff} exceeds tolerance at q={q}"

    def test_sensor_transformation_tree(self, solver, sim):
        """Verify camera and ToF sensor frame positions against MuJoCo sensor sites."""
        q_home = solver.HOME_POSITION
        sim.set_joint_positions(q_home)

        sensor_poses = solver.get_sensor_poses(q_home)
        assert "camera" in sensor_poses
        assert "tof" in sensor_poses
        assert "forearm" in sensor_poses

        # Compare camera position
        sim_cam_pos = sim.data.cam_xpos[sim.camera.cam_id]
        ikpy_cam_pos = sensor_poses["camera"][0]
        assert np.linalg.norm(sim_cam_pos - ikpy_cam_pos) < 1e-4

        # Compare ToF position
        sim_tof_pos = sim.data.site_xpos[sim.tof.site_id]
        ikpy_tof_pos = sensor_poses["tof"][0]
        assert np.linalg.norm(sim_tof_pos - ikpy_tof_pos) < 1e-4


class TestReachabilityAndSingularity:
    """Test reachability envelope, table grid coverage, and kinematic singularity detection."""

    @pytest.fixture
    def solver(self):
        return KinematicsSolver()

    def test_reachability_workspace(self, solver):
        """Verify reachability check distinguishes valid table locations from out-of-bound coordinates."""
        # Realistic table coordinates in workspace
        reachable_points = [
            [0.28, 0.05, 0.03],
            [0.32, 0.16, 0.025],
            [0.30, -0.12, 0.02],
            [0.25, 0.0, 0.05],
            [0.35, 0.10, 0.10],
        ]
        for pt in reachable_points:
            assert solver.is_reachable(pt), f"Expected point {pt} to be reachable"

        # Clearly unreachable points
        unreachable_points = [
            [0.85, 0.0, 0.0],     # Too far in X
            [0.0, 0.70, 0.0],     # Too far in Y
            [0.089, 0.0, 0.0],    # Inside base stand (radius < 0.12m)
            [0.30, 0.0, -0.30],   # Deep underneath table
            [0.30, 0.0, 0.80],    # Far above robot ceiling
        ]
        for pt in unreachable_points:
            assert not solver.is_reachable(pt), f"Expected point {pt} to be unreachable"

    def test_reachability_grid_computation(self, solver):
        """Compute reachability grid on table and assert high fraction of reachable table surface."""
        grid_data = solver.compute_reachability_grid(
            x_bounds=(0.22, 0.38),
            y_bounds=(-0.15, 0.15),
            z_level=0.025,
            resolution=0.04,
        )
        assert grid_data["reachable_fraction"] > 0.75, "Expected >=75% reachability on primary work area"

    def test_singularity_detection(self, solver):
        """Verify Yoshikawa manipulability index drops at singularities and is healthy at home."""
        q_home = solver.HOME_POSITION
        mu_home = solver.manipulability_index(q_home)
        assert mu_home > 0.005, f"Home manipulability {mu_home} should be healthy"
        assert not solver.is_singular(q_home)

        # Straightened singular arm posture [0, 0, 0, 0, 0]
        q_straight = np.zeros(5)
        mu_straight = solver.manipulability_index(q_straight)
        assert mu_straight < 1e-4, f"Straight posture manipulability {mu_straight} should be near zero"
        assert solver.is_singular(q_straight, threshold=1e-4)

        # Condition number check
        cond_home = solver.condition_number(q_home)
        assert cond_home < 500.0


class TestInverseKinematics:
    """Test Inverse Kinematics accuracy (< 2 mm), orientation modes, and multi-seed selection."""

    @pytest.fixture
    def solver(self):
        return KinematicsSolver()

    def test_ik_accuracy_grid_on_table(self, solver):
        """Test IK accuracy on a coordinate grid across the table surface with < 2 mm residual."""
        test_grid = [
            [0.24, 0.0, 0.02],
            [0.26, 0.10, 0.025],
            [0.28, -0.08, 0.02],
            [0.30, 0.05, 0.03],
            [0.32, -0.12, 0.02],
            [0.34, 0.14, 0.025],
        ]

        for target in test_grid:
            res = solver.solve_ik(target, orientation_mode="auto", pos_tol_mm=2.0)
            assert res.success, f"Failed IK at target {target}: {res.message}"
            assert res.position_error_mm < 2.0, (
                f"Position residual {res.position_error_mm:.3f} mm exceeded 2.0 mm at {target}"
            )
            # Verify limits
            assert solver.limits.is_valid(res.positions)

    def test_ik_orientation_modes(self, solver):
        """Verify top_down, slanted, free, and auto orientation modes."""
        target = [0.28, 0.05, 0.02]

        # 1. Top-down: End-effector Z-axis must align with -Z (downward)
        res_td = solver.solve_ik(target, orientation_mode="top_down")
        assert res_td.success
        assert res_td.position_error_mm < 2.0
        fk_td = solver.compute_fk(res_td.positions)
        z_axis = fk_td[:3, 2]
        # Dot product with [0, 0, -1] should be close to 1
        dot_td = np.dot(z_axis, [0.0, 0.0, -1.0])
        assert dot_td > 0.98, f"Top-down Z-axis dot product {dot_td} should be > 0.98"

        # 2. Free orientation: Prioritizes exact position
        res_free = solver.solve_ik(target, orientation_mode="free")
        assert res_free.success
        assert res_free.position_error_mm < 0.1

        # 3. Slanted approach
        res_slanted = solver.solve_ik(target, orientation_mode="slanted")
        assert res_slanted.success
        assert res_slanted.position_error_mm < 2.0

        # 4. Auto orientation: Selects best achievable orientation
        res_auto = solver.solve_ik(target, orientation_mode="auto")
        assert res_auto.success
        assert res_auto.position_error_mm < 2.0

    def test_minimal_joint_displacement(self, solver):
        """Ensure solver chooses solution closest to initial posture when multiple exist."""
        target = [0.28, 0.05, 0.02]
        q_init = solver.HOME_POSITION.copy()

        res = solver.solve_ik(target, initial_position=q_init, orientation_mode="auto")
        assert res.success
        displacement = np.linalg.norm(res.positions - q_init)
        # Should smoothly adjust joints without flipping 180 degrees
        assert displacement < 2.0, f"Displacement {displacement:.2f} rad is excessively large"

    def test_unreachable_target_handling(self, solver):
        """Unreachable target outside workspace returns success=False without crashing."""
        far_target = [1.5, 2.0, 0.0]
        res = solver.solve_ik(far_target)
        assert not res.success
        assert "outside reachability workspace" in res.message


class TestTrajectoryGenerationAndSimulation:
    """Test S-curve, trapezoidal profile, waypoint pipeline, and physics execution in MuJoCo."""

    @pytest.fixture
    def solver(self):
        return KinematicsSolver()

    @pytest.fixture
    def generator(self):
        return TrajectoryGenerator()

    @pytest.fixture
    def sim(self):
        s = MujocoSim()
        yield s
        s.close()

    def test_s_curve_continuity_and_limits(self, generator, solver):
        """Verify S-curve profile satisfies boundary conditions and velocity/acceleration limits."""
        q0 = solver.HOME_POSITION
        q1 = np.array([-0.3, 0.4, 1.1, 0.9, 0.2])

        traj = generator.plan_segment(q0, q1, profile="s_curve", dt=0.01)
        assert len(traj) > 10

        # Boundary checks
        assert np.allclose(traj[0].positions, q0, atol=1e-4)
        assert np.allclose(traj[-1].positions, q1, atol=1e-4)

        # C2 boundary condition: initial and final velocities and accelerations are zero
        assert np.allclose(traj[0].velocities, np.zeros(5), atol=1e-4)
        assert np.allclose(traj[-1].velocities, np.zeros(5), atol=1e-4)
        assert np.allclose(traj[0].accelerations, np.zeros(5), atol=1e-4)
        assert np.allclose(traj[-1].accelerations, np.zeros(5), atol=1e-4)

        # Velocity and acceleration limits respected throughout
        max_vel = generator.limits.max_velocity
        max_acc = generator.limits.max_acceleration

        for p in traj:
            assert np.all(np.abs(p.velocities) <= max_vel + 1e-4)
            assert np.all(np.abs(p.accelerations) <= max_acc + 1e-4)

    def test_trapezoidal_profile(self, generator, solver):
        """Verify trapezoidal profile reaches target configuration within limits."""
        q0 = solver.HOME_POSITION
        q1 = np.array([0.2, 0.5, 0.8, 0.7, -0.2])

        traj = generator.plan_segment(q0, q1, profile="trapezoidal", dt=0.01)
        assert len(traj) > 10
        assert np.allclose(traj[0].positions, q0, atol=1e-4)
        assert np.allclose(traj[-1].positions, q1, atol=1e-4)

        max_vel = generator.limits.max_velocity
        for p in traj:
            assert np.all(np.abs(p.velocities) <= max_vel + 1e-3)

    def test_pick_and_place_waypoint_pipeline(self, generator, solver):
        """Verify full waypoint pipeline generates all key pick-and-place stages."""
        pick_pos = [0.28, 0.05, 0.025]
        drop_pos = [0.18, 0.20, 0.035]

        waypoints, trajectory = generator.plan_pick_and_place(
            pick_position=pick_pos,
            dropoff_position=drop_pos,
            solver=solver,
        )

        expected_stages = [
            "HOME",
            "PRE_GRASP",
            "GRASP",
            "LIFT",
            "TRANSFER",
            "PLACE",
            "POST_PLACE",
            "RETURN_HOME",
        ]
        actual_stages = [w.name for w in waypoints]
        assert actual_stages == expected_stages

        # Trajectory timestamps must be strictly monotonic
        times = [p.time for p in trajectory]
        for i in range(len(times) - 1):
            assert times[i + 1] >= times[i]

        # Gripper opening checks
        assert waypoints[0].gripper == 0.04  # Open at Home
        assert waypoints[1].gripper == 0.04  # Open at Pre-grasp
        assert waypoints[2].gripper == 0.00  # Closed at Grasp
        assert waypoints[3].gripper == 0.00  # Closed at Lift
        assert waypoints[5].gripper == 0.04  # Open at Place

    def test_mujoco_trajectory_tracking_smoothness(self, generator, solver, sim):
        """Execute S-curve segment in MuJoCo physics engine and assess tracking and smoothness."""
        q_start = solver.HOME_POSITION
        # Target position on table
        ik_target = solver.solve_ik([0.29, 0.06, 0.025], orientation_mode="auto")
        assert ik_target.success
        q_goal = ik_target.positions

        sim.set_joint_positions(q_start)
        sim.set_joint_targets(q_start)
        sim.step(30)

        # Generate S-curve trajectory
        traj = generator.plan_segment(q_start, q_goal, profile="s_curve", dt=0.01)

        sim_dt = sim.model.opt.timestep  # e.g. 0.002s
        steps_per_point = max(1, int(round(0.01 / sim_dt)))

        for pt in traj:
            sim.set_joint_targets(pt.positions)
            sim.step(steps_per_point)

        # Allow brief settling
        sim.step(50)

        # Check final position tracking accuracy in physics (allowing for P-controller gravity droop ~0.09 rad)
        final_joint_state = sim.get_joint_states()
        final_q = np.array(final_joint_state.positions)
        joint_tracking_error = np.linalg.norm(final_q - q_goal)
        assert joint_tracking_error < 0.12, f"Joint error {joint_tracking_error} rad exceeded tolerance"

        # Check final TCP position error in MuJoCo
        tcp_pos = sim.data.site_xpos[sim.tcp_site_id]
        tcp_err_mm = np.linalg.norm(tcp_pos - [0.29, 0.06, 0.025]) * 1000.0
        assert tcp_err_mm < 45.0, f"MuJoCo TCP tracking error {tcp_err_mm:.2f} mm exceeded 45 mm"
