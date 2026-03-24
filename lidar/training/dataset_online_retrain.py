#!/usr/bin/env python3
"""Sample-manifest based LiDAR dataset for online retraining."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


def resample_points(points: np.ndarray, target_num_points: int) -> np.ndarray:
    if points.shape[0] == target_num_points:
        return points.astype(np.float32)
    if points.shape[0] == 0:
        return np.zeros((target_num_points, 3), dtype=np.float32)
    if points.shape[0] > target_num_points:
        indices = np.random.choice(points.shape[0], target_num_points, replace=False)
        return points[indices].astype(np.float32)
    indices = np.random.choice(points.shape[0], target_num_points - points.shape[0], replace=True)
    return np.vstack([points, points[indices]]).astype(np.float32)


def compute_reference_frame(sequence: np.ndarray) -> Tuple[np.ndarray, float]:
    for frame in sequence:
        valid = frame[np.linalg.norm(frame, axis=1) > 0]
        if len(valid) == 0:
            continue

        # Use the earliest viable frame in the 48-frame window so training
        # matches online inference more closely, but avoid degenerate frames
        # that contain only one repeated point after ROI resampling.
        unique_points = np.unique(valid, axis=0)
        if len(unique_points) < 4:
            continue

        centroid = unique_points.mean(axis=0)
        distances = np.linalg.norm(unique_points - centroid, axis=1)
        scale = float(np.percentile(distances, 95)) if len(distances) > 0 else 0.0
        if scale >= 1.0e-2:
            return centroid.astype(np.float32), scale

    return np.zeros(3, dtype=np.float32), 1.0


def normalize_sequence_globally(sequence: np.ndarray) -> np.ndarray:
    centroid, scale = compute_reference_frame(sequence)
    return ((sequence - centroid) / scale).astype(np.float32)


class SequenceAugmentation:
    def __init__(self, rotation_range: float = 0.1, scale_range: Tuple[float, float] = (0.9, 1.1),
                 jitter_sigma: float = 0.01, jitter_clip: float = 0.05):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        scale = np.random.uniform(*self.scale_range)
        jitter = np.clip(
            self.jitter_sigma * np.random.randn(*sequence.shape),
            -self.jitter_clip,
            self.jitter_clip,
        ).astype(np.float32)
        return sequence @ rotation_matrix.T * scale + jitter


class OnlineRetrainLiDARDataset(Dataset):
    def __init__(self, manifest_path: str | Path, frame_root: str | Path, subjects: List[str] | None = None,
                 clip_len: int = 48, target_num_points: int = 256, augment: bool = False,
                 oversample_config: Dict[str, int] | None = None):
        self.manifest_path = Path(manifest_path)
        self.frame_root = Path(frame_root)
        self.subjects = set(subjects or [])
        self.clip_len = clip_len
        self.target_num_points = target_num_points
        self.augment = SequenceAugmentation() if augment else None
        self.oversample_config = oversample_config or {}

        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        samples = manifest['samples'] if isinstance(manifest, dict) else manifest
        if self.subjects:
            samples = [sample for sample in samples if sample['subject'] in self.subjects]

        self.samples = []
        for sample in samples:
            repeat = self.oversample_config.get(sample.get('lighting_level', 'normal'), 1) if self.oversample_config else 1
            for _ in range(repeat):
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        trial_dir = self.frame_root / sample['trial_name']
        sequence = []
        for frame_number in sample['frame_numbers']:
            frame_path = trial_dir / f'{frame_number:05d}.npy'
            points = np.load(frame_path) if frame_path.exists() else np.zeros((0, 3), dtype=np.float32)
            sequence.append(resample_points(points, self.target_num_points))
        sequence = normalize_sequence_globally(np.asarray(sequence, dtype=np.float32))
        if self.augment is not None:
            sequence = self.augment(sequence)
        metadata = {
            'sample_uid': sample['sample_uid'],
            'trial_name': sample['trial_name'],
            'subject': sample['subject'],
            'environment': sample['environment'],
            'activity': sample['activity'],
            'lighting_level': sample['lighting_level'],
            'window': [sample['start_frame'], sample['end_frame']],
        }
        return torch.FloatTensor(sequence), int(sample['label']), metadata
