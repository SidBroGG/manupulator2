import os
import pytest

def test_core_imports():
    """Verify all core dependencies import cleanly."""
    import zenoh
    import cv2
    import mediapipe
    import ultralytics
    import ikpy
    import mujoco
    import numpy as np
    import scipy
    import yaml
    import pydantic

    assert np.__version__ is not None
    assert mujoco.__version__ is not None

def test_urdf_model_exists():
    """Verify URDF model file is present in urdf_model directory."""
    urdf_path = os.path.join("urdf_model", "root_mujoco.urdf")
    assert os.path.exists(urdf_path), f"URDF file not found at {urdf_path}"
    assert os.path.getsize(urdf_path) > 0, "URDF file is empty"
