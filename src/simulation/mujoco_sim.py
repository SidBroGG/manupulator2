"""MuJoCo Simulation environment and Digital Twin for the 6-DOF manipulator."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import yaml
import numpy as np
import mujoco

from src.core.types import (
    JointState,
    JointCommand,
    CameraFrame,
    ToFDistance,
    PointingGesture,
)
from src.simulation.virtual_sensors import (
    VirtualCamera,
    VirtualToF,
    PointingGestureInjector,
)


class MujocoSim:
    """Manages the MuJoCo simulation environment, digital twin, actuators, and sensors."""

    ARM_JOINT_NAMES = [
        "revolute_1",
        "revolute_2",
        "revolute_3",
        "revolute_4",
        "revolute_5",
    ]

    MANIPULABLE_OBJECTS = ["red_box", "blue_cylinder", "green_sphere"]

    def __init__(
        self,
        config_path: str = "config/default_config.yaml",
        scene_path: Optional[str] = None,
    ) -> None:
        self.config = self._load_config(config_path)

        if scene_path is None:
            # Resolve default scene relative to repository root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scene_path = os.path.join(base_dir, "simulation", "assets", "manipulator_scene.xml")

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Simulation scene file not found at: {scene_path}")

        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Set physics timestep from config
        sim_cfg = self.config.get("simulation", {})
        if "physics_timestep" in sim_cfg:
            self.model.opt.timestep = float(sim_cfg["physics_timestep"])

        # Forward kinematics initialization
        mujoco.mj_forward(self.model, self.data)

        # Index arm joints
        self.joint_ids = {}
        self.joint_qpos_adr = {}
        self.joint_dof_adr = {}
        for name in self.ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Arm joint '{name}' not found in scene XML.")
            self.joint_ids[name] = jid
            self.joint_qpos_adr[name] = self.model.jnt_qposadr[jid]
            self.joint_dof_adr[name] = self.model.jnt_dofadr[jid]

        # Index actuators
        self.actuator_ids = {}
        for i in range(self.model.nu):
            act_name = self.model.actuator(i).name
            self.actuator_ids[act_name] = i

        # Index manipulable objects and welds
        self.object_joints = {}
        self.object_bodies = {}
        self.weld_equality_ids = {}

        for obj_name in self.MANIPULABLE_OBJECTS:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{obj_name}")
            wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{obj_name}")

            if bid >= 0:
                self.object_bodies[obj_name] = bid
            if jid >= 0:
                self.object_joints[obj_name] = jid
            if wid >= 0:
                self.weld_equality_ids[obj_name] = wid

        # Gripper TCP site
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")

        # Initialize virtual sensors
        self.camera = VirtualCamera(self.model, self.data, camera_name="forearm_camera")
        self.tof = VirtualToF(self.model, self.data, site_name="tof_sensor_site", camera_name="forearm_camera")
        self.gesture_injector = PointingGestureInjector(self.model, self.data, table_z=0.0)

        # Currently grasped object
        self._grasped_object: Optional[str] = None

        # Set default home pose
        self.reset_home()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def reset_home(self) -> None:
        """Move arm joints to configured home positions."""
        joints_cfg = self.config.get("robot", {}).get("joints", [])
        home_positions = []
        for j_cfg in joints_cfg:
            home_positions.append(float(j_cfg.get("home_pos", 0.0)))

        if len(home_positions) == len(self.ARM_JOINT_NAMES):
            self.set_joint_positions(home_positions)
            self.set_joint_targets(home_positions)
        self.set_gripper(0.04)  # Open gripper by default

    def step(self, n_steps: int = 1) -> None:
        """Advance the physics simulation by n_steps."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def set_joint_positions(self, positions: Union[List[float], Dict[str, float]]) -> None:
        """Directly set joint angles (kinematic teleoperation / reset)."""
        if isinstance(positions, dict):
            for name, val in positions.items():
                if name in self.joint_qpos_adr:
                    self.data.qpos[self.joint_qpos_adr[name]] = val
                    self.data.qvel[self.joint_dof_adr[name]] = 0.0
        else:
            for i, val in enumerate(positions[: len(self.ARM_JOINT_NAMES)]):
                name = self.ARM_JOINT_NAMES[i]
                self.data.qpos[self.joint_qpos_adr[name]] = val
                self.data.qvel[self.joint_dof_adr[name]] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, targets: Union[List[float], Dict[str, float]]) -> None:
        """Command position actuators to desired target angles in radians."""
        if isinstance(targets, dict):
            for name, val in targets.items():
                act_name = f"actuator_{name}"
                if act_name in self.actuator_ids:
                    self.data.ctrl[self.actuator_ids[act_name]] = val
        else:
            for i, val in enumerate(targets[: len(self.ARM_JOINT_NAMES)]):
                name = self.ARM_JOINT_NAMES[i]
                act_name = f"actuator_{name}"
                if act_name in self.actuator_ids:
                    self.data.ctrl[self.actuator_ids[act_name]] = val

    def get_joint_states(self) -> JointState:
        """Get current positions, velocities, and efforts of all 5 arm joints."""
        positions = [float(self.data.qpos[self.joint_qpos_adr[name]]) for name in self.ARM_JOINT_NAMES]
        velocities = [float(self.data.qvel[self.joint_dof_adr[name]]) for name in self.ARM_JOINT_NAMES]
        efforts = []
        for name in self.ARM_JOINT_NAMES:
            act_name = f"actuator_{name}"
            if act_name in self.actuator_ids:
                efforts.append(float(self.data.actuator_force[self.actuator_ids[act_name]]))
            else:
                efforts.append(0.0)

        return JointState(
            positions=positions,
            velocities=velocities,
            effort=efforts,
            timestamp=float(self.data.time),
        )

    def set_gripper(self, open_amount_m: float) -> None:
        """Control gripper opening. open_amount_m in [0.0, 0.04] (0=closed, 0.04=fully open)."""
        open_amount_m = np.clip(open_amount_m, 0.0, 0.04)
        # Linear map: open_amount=0.04 -> ctrl=-0.015 (open); open_amount=0.0 -> ctrl=0.005 (closed)
        ctrl_val = float(-0.015 + (1.0 - open_amount_m / 0.04) * 0.020)
        if "actuator_gripper" in self.actuator_ids:
            self.data.ctrl[self.actuator_ids["actuator_gripper"]] = ctrl_val

    def grasp_object(self, object_name: Optional[str] = None) -> bool:
        """Close gripper and activate weld equality constraint if object is within grasp radius."""
        self.set_gripper(0.0)  # Close physical fingers
        self.step(20)

        tcp_pos = self.data.site_xpos[self.tcp_site_id] if self.tcp_site_id >= 0 else np.array([0, 0, 0])

        target_obj = object_name
        if target_obj is None:
            # Find closest object
            min_dist = float("inf")
            for obj in self.MANIPULABLE_OBJECTS:
                if obj in self.object_bodies:
                    obj_pos = self.data.xpos[self.object_bodies[obj]]
                    dist = float(np.linalg.norm(tcp_pos - obj_pos))
                    if dist < min_dist:
                        min_dist = dist
                        target_obj = obj

        if target_obj and target_obj in self.weld_equality_ids:
            obj_pos = self.data.xpos[self.object_bodies[target_obj]]
            dist = float(np.linalg.norm(tcp_pos - obj_pos))
            # If within 7cm of TCP, securely weld object to gripper
            if dist < 0.07:
                wid = self.weld_equality_ids[target_obj]
                self.data.eq_active[wid] = True
                self._grasped_object = target_obj
                return True

        return False

    def release_object(self) -> None:
        """Open gripper and release any grasped object."""
        for wid in self.weld_equality_ids.values():
            self.data.eq_active[wid] = False
        self._grasped_object = None
        self.set_gripper(0.04)
        self.step(10)

    def reset_objects(
        self,
        positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        randomize: bool = False,
    ) -> None:
        """Reset object locations on the workspace table."""
        self.release_object()

        defaults = {
            "red_box": (0.28, 0.05, 0.02),
            "blue_cylinder": (0.32, 0.16, 0.025),
            "green_sphere": (0.30, -0.12, 0.02),
        }

        for obj_name, jid in self.object_joints.items():
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]

            if positions and obj_name in positions:
                pos = positions[obj_name]
            elif randomize:
                # Randomize within reachable table area (X in [0.22, 0.38], Y in [-0.20, 0.20])
                rx = float(np.random.uniform(0.22, 0.38))
                ry = float(np.random.uniform(-0.20, 0.20))
                pos = (rx, ry, defaults[obj_name][2])
            else:
                pos = defaults.get(obj_name, (0.3, 0.0, 0.02))

            self.data.qpos[qadr : qadr + 3] = pos
            self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]  # Unit quaternion
            self.data.qvel[dadr : dadr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def get_object_position(self, object_name: str) -> np.ndarray:
        """Get 3D position (x, y, z) of an object in world coordinates."""
        if object_name in self.object_bodies:
            return self.data.xpos[self.object_bodies[object_name]].copy()
        raise ValueError(f"Unknown object: {object_name}")

    def get_camera_frame(self) -> CameraFrame:
        """Capture RGB frame from forearm camera."""
        return self.camera.capture_frame()

    def get_tof_distance(self) -> ToFDistance:
        """Measure distance from forearm ToF sensor."""
        return self.tof.measure_distance()

    def inject_gesture(
        self,
        target_xy: Tuple[float, float],
        origin_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> PointingGesture:
        """Simulate a human pointing gesture targeting a specific table coordinate."""
        gesture = self.gesture_injector.create_gesture(target_xy, origin_xyz)
        mujoco.mj_forward(self.model, self.data)
        return gesture

    def close(self) -> None:
        """Clean up simulator and sensor resources."""
        self.camera.close()
