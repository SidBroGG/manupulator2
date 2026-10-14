"""Core data models and message schemas for the manipulator system."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class JointState(BaseModel):
    """Represents the current joint positions, velocities, and efforts."""
    positions: List[float] = Field(description="Joint positions in radians or meters")
    velocities: Optional[List[float]] = Field(default=None, description="Joint velocities in rad/s")
    effort: Optional[List[float]] = Field(default=None, description="Joint torques/forces")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")


class JointCommand(BaseModel):
    """Command sent to joint actuators."""
    positions: List[float] = Field(description="Target joint angles in radians")
    gripper: Optional[float] = Field(default=None, description="Gripper opening in meters (0=closed)")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")


class CameraFrame(BaseModel):
    """Metadata and raw RGB array wrapper for forearm camera."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    width: int = 640
    height: int = 480
    fps: int = 30
    timestamp: float = 0.0
    image: Optional[Any] = Field(default=None, description="numpy.ndarray RGB image")


class ToFDistance(BaseModel):
    """Reading from VL53L0X ToF rangefinder."""
    distance_m: float = Field(description="Measured distance in meters")
    valid: bool = Field(default=True, description="True if distance is within valid sensor range")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")


class Ray3D(BaseModel):
    """3D ray represented by an origin point and a unit direction vector."""
    origin: Tuple[float, float, float] = Field(description="(x, y, z) origin")
    direction: Tuple[float, float, float] = Field(description="(dx, dy, dz) unit vector")


class PointingGesture(BaseModel):
    """Human pointing gesture reconstructed in 3D space."""
    ray: Ray3D
    table_intersection: Optional[Tuple[float, float, float]] = Field(
        default=None, description="Point (x, y, z) where ray intersects table plane"
    )
    confidence: float = Field(default=1.0, description="Confidence score [0..1]")
    timestamp: float = 0.0


class SimulatedObject(BaseModel):
    """Information about an object in the simulation workspace."""
    name: str
    shape: str = Field(description="'box', 'cylinder', 'sphere'")
    position: Tuple[float, float, float] = Field(description="(x, y, z) world coordinates")
    orientation: Tuple[float, float, float, float] = Field(
        default=(1.0, 0.0, 0.0, 0.0), description="Quaternion (w, x, y, z)"
    )
    size: Tuple[float, ...] = Field(description="Dimensions (half-lengths for box, radius/half-height for cylinder)")
    mass: float = Field(default=0.05, description="Mass in kg")
    color_rgba: Tuple[float, float, float, float] = Field(default=(1.0, 0.0, 0.0, 1.0))
