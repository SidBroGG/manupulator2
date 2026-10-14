"""Comprehensive unit and integration tests for Stage 1: MuJoCo Simulation and Digital Twin."""

import os
import numpy as np
import pytest
import mujoco

from src.simulation.mujoco_sim import MujocoSim
from src.simulation.virtual_sensors import (
    VirtualCamera,
    VirtualToF,
    PointingGestureInjector,
)
from src.core.types import JointState, CameraFrame, ToFDistance, PointingGesture


@pytest.fixture
def sim():
    """Fixture providing an initialized MujocoSim instance."""
    simulator = MujocoSim(
        config_path="config/default_config.yaml",
        scene_path="src/simulation/assets/manipulator_scene.xml",
    )
    yield simulator
    simulator.close()


def test_urdf_compilation_in_mujoco():
    """Verify that urdf_model/root_mujoco.urdf compiles cleanly in MuJoCo (Item 1.1.1)."""
    urdf_path = os.path.abspath("urdf_model/root_mujoco.urdf")
    assert os.path.exists(urdf_path)
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)
    assert model.nbody == 6
    assert model.njnt == 5

    # Run step without errors
    mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()


def test_master_scene_structure(sim: MujocoSim):
    """Verify master scene loads joints, actuators, camera, sites, and objects (Item 1.1, 1.2)."""
    # 5 arm joints + 2 gripper slide joints + 3 freejoints for objects = 10 joints
    assert sim.model.njnt >= 7
    # 5 arm actuators + 1 gripper actuator = 6 actuators
    assert sim.model.nu == 6

    # Verify joint names
    for name in sim.ARM_JOINT_NAMES:
        assert name in sim.joint_ids

    # Verify actuators
    for name in sim.ARM_JOINT_NAMES:
        assert f"actuator_{name}" in sim.actuator_ids
    assert "actuator_gripper" in sim.actuator_ids

    # Verify table surface exists and is at Z = 0.0
    table_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface")
    assert table_geom_id >= 0
    # Center is at Z = -0.01 with half-height 0.01 -> top surface is Z = 0.0
    assert np.isclose(sim.model.geom_pos[table_geom_id][2] + sim.model.geom_size[table_geom_id][2], 0.0)


def test_physics_stability_and_no_explosions(sim: MujocoSim):
    """Test 500 physics simulation steps without explosions or NaNs (Item 1.4.1)."""
    sim.reset_home()
    for _ in range(500):
        sim.step(1)
        assert not np.isnan(sim.data.qpos).any()
        assert not np.isnan(sim.data.qvel).any()
        # Verify robot parts don't fly off (positions within reasonable workspace)
        for i in range(sim.model.nbody):
            assert np.linalg.norm(sim.data.xpos[i]) < 5.0, f"Body {i} exploded: {sim.data.xpos[i]}"


def test_actuator_control_and_joint_limits(sim: MujocoSim):
    """Test commanding joint targets and verify actuator response (Item 1.1.3)."""
    sim.reset_home()
    target_angles = [0.3, 0.4, 0.4, 0.5, 0.2]
    sim.set_joint_targets(target_angles)

    # Step simulation to let position controllers drive the joints
    sim.step(250)

    states: JointState = sim.get_joint_states()
    for commanded, reached in zip(target_angles, states.positions):
        # Allow small settling tolerance under gravity
        assert np.isclose(commanded, reached, atol=0.08), f"Commanded {commanded}, got {reached}"


def test_virtual_camera_rendering(sim: MujocoSim):
    """Test synthetic video frame capture from forearm camera (Item 1.3.1, 1.4.2)."""
    frame: CameraFrame = sim.get_camera_frame()
    assert isinstance(frame, CameraFrame)
    assert frame.width == 640
    assert frame.height == 480
    assert frame.image is not None
    assert frame.image.shape == (480, 640, 3)
    assert frame.image.dtype == np.uint8

    # Frame should contain visual data (not completely black)
    assert frame.image.mean() > 5.0

    # Depth rendering test
    depth = sim.camera.capture_depth()
    assert depth.shape == (480, 640)
    assert depth.min() > 0.0
    # Foreground table/scene depth exists in camera view
    assert np.any(depth < 2.0)


def test_virtual_tof_sensor_measurements(sim: MujocoSim):
    """Test multi-ray ToF sensor range measurements towards table and objects (Item 1.3.2, 1.4.2)."""
    # At approach / observation pose, claw and ToF point towards table
    sim.set_joint_positions([0.0, 0.8, 0.8, 0.6, 0.0])
    sim.step(10)

    reading: ToFDistance = sim.get_tof_distance()
    assert isinstance(reading, ToFDistance)
    assert reading.valid
    # Claw is close to table surface
    assert 0.03 <= reading.distance_m <= 0.50

    # Test out of range behavior when pointing vertically up towards ceiling/sky
    sim.set_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0])
    sim.step(10)
    reading_sky = sim.get_tof_distance()
    assert reading_sky.distance_m == sim.tof.max_range_m
    assert not reading_sky.valid


def test_pointing_gesture_injector(sim: MujocoSim):
    """Test 3D pointing ray generation and table intersection calculation (Item 1.3.3)."""
    target_xy = (0.35, 0.10)
    gesture: PointingGesture = sim.inject_gesture(target_xy)

    assert isinstance(gesture, PointingGesture)
    assert gesture.table_intersection is not None
    inter_x, inter_y, inter_z = gesture.table_intersection

    # Intersection Z must match table surface (0.0)
    assert np.isclose(inter_z, 0.0, atol=1e-4)
    assert np.isclose(inter_x, target_xy[0], atol=1e-3)
    assert np.isclose(inter_y, target_xy[1], atol=1e-3)

    # Ray direction must be unit vector
    dir_vec = np.array(gesture.ray.direction)
    assert np.isclose(np.linalg.norm(dir_vec), 1.0, atol=1e-5)


def test_object_manipulation_and_grasping(sim: MujocoSim):
    """Test object resetting, physical gripper closure, and reliable weld constraint (Item 1.2.2, 1.2.3)."""
    # 1. Reset object positions
    custom_pos = {"red_box": (0.30, 0.0, 0.02)}
    sim.reset_objects(positions=custom_pos)
    red_pos = sim.get_object_position("red_box")
    assert np.isclose(red_pos[0], 0.30, atol=0.01)
    assert np.isclose(red_pos[1], 0.0, atol=0.01)

    # 2. Test random object reset within workspace
    sim.reset_objects(randomize=True)
    for obj in sim.MANIPULABLE_OBJECTS:
        pos = sim.get_object_position(obj)
        assert 0.15 <= pos[0] <= 0.45
        assert -0.25 <= pos[1] <= 0.25

    # 3. Test grasp and release with object at gripper TCP
    sim.reset_home()
    sim.step(10)
    tcp_pos = sim.data.site_xpos[sim.tcp_site_id].copy()
    sim.reset_objects(positions={"red_box": tuple(tcp_pos)})
    sim.step(10)

    # Attempt grasp
    grasped = sim.grasp_object("red_box")
    assert grasped
    assert sim._grasped_object == "red_box"
    weld_id = sim.weld_equality_ids["red_box"]
    assert sim.data.eq_active[weld_id]

    # Release
    sim.release_object()
    assert sim._grasped_object is None
    assert not sim.data.eq_active[weld_id]
