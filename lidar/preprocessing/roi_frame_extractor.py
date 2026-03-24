"""ROI-only LiDAR preprocessing helpers for online_train."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def create_rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    rx = np.radians(angle_x)
    ry = np.radians(angle_y)
    rz = np.radians(angle_z)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return Rx @ Ry @ Rz


@dataclass
class ManualROI:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_dict(cls, data: Dict[str, list[float]]) -> 'ManualROI':
        return cls(
            x_min=float(data['x'][0]),
            x_max=float(data['x'][1]),
            y_min=float(data['y'][0]),
            y_max=float(data['y'][1]),
            z_min=float(data['z'][0]),
            z_max=float(data['z'][1]),
        )

    def to_dict(self) -> Dict[str, list[float]]:
        return {
            'x': [self.x_min, self.x_max],
            'y': [self.y_min, self.y_max],
            'z': [self.z_min, self.z_max],
        }


def apply_roi(points: np.ndarray, roi: ManualROI) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    mask = (
        (roi.x_min <= points[:, 0]) & (points[:, 0] <= roi.x_max) &
        (roi.y_min <= points[:, 1]) & (points[:, 1] <= roi.y_max) &
        (roi.z_min <= points[:, 2]) & (points[:, 2] <= roi.z_max)
    )
    return points[mask].astype(np.float32)


class ROIFrameExtractor:
    def __init__(self, rotation_degrees: Dict[str, float], roi: ManualROI, target_num_points: int = 256) -> None:
        self.rotation_degrees = {axis: float(rotation_degrees.get(axis, 0.0)) for axis in ('x', 'y', 'z')}
        self.rotation_matrix = create_rotation_matrix(
            self.rotation_degrees['x'],
            self.rotation_degrees['y'],
            self.rotation_degrees['z'],
        )
        self.roi = roi
        self.target_num_points = int(target_num_points)

    def rotate(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return (points.astype(np.float32) @ self.rotation_matrix.T).astype(np.float32)

    def crop(self, points: np.ndarray) -> np.ndarray:
        return apply_roi(points.astype(np.float32), self.roi)

    def resample(self, points: np.ndarray) -> np.ndarray:
        if self.target_num_points <= 0:
            return points.astype(np.float32)
        count = len(points)
        if count == 0:
            return np.zeros((self.target_num_points, 3), dtype=np.float32)
        if count >= self.target_num_points:
            indices = np.random.choice(count, self.target_num_points, replace=False)
            return points[indices].astype(np.float32)
        indices = np.random.choice(count, self.target_num_points - count, replace=True)
        return np.vstack([points, points[indices]]).astype(np.float32)

    def process_points(self, raw_points: np.ndarray) -> np.ndarray:
        rotated = self.rotate(raw_points)
        cropped = self.crop(rotated)
        return self.resample(cropped)
