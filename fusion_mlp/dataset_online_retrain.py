#!/usr/bin/env python3
"""Fusion dataset for online retraining."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from online_train.lidar.training.dataset_online_retrain import normalize_sequence_globally, resample_points


class OnlineRetrainFusionPoseDataset(Dataset):
    def __init__(self, pose_dataset: Dataset, lidar_manifest_path: str | Path, pointcloud_data_dir: str | Path,
                 target_num_points: int = 256):
        self.pose_dataset = pose_dataset
        self.pointcloud_data_dir = Path(pointcloud_data_dir)
        self.target_num_points = target_num_points
        with open(lidar_manifest_path, 'r') as f:
            manifest = json.load(f)
        samples = manifest['samples'] if isinstance(manifest, dict) else manifest
        self.lidar_by_uid = {sample['sample_uid']: sample for sample in samples}
        self.annotations = self._filter_valid_samples()

    def _filter_valid_samples(self):
        annotations = []
        for idx in range(len(self.pose_dataset)):
            pose_ann = self.pose_dataset.get_data_info(idx)
            sample_uid = pose_ann.get('sample_uid')
            if not sample_uid:
                continue
            lidar_ann = self.lidar_by_uid.get(sample_uid)
            if lidar_ann is None:
                continue
            annotations.append({'pose_idx': idx, 'lidar_ann': lidar_ann})
        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        pose_data = self.pose_dataset[ann['pose_idx']]
        lidar_ann = ann['lidar_ann']
        trial_dir = self.pointcloud_data_dir / lidar_ann['trial_name']

        frames = []
        for frame_number in lidar_ann['frame_numbers']:
            frame_path = trial_dir / f'{frame_number:05d}.npy'
            points = np.load(frame_path) if frame_path.exists() else np.zeros((0, 3), dtype=np.float32)
            frames.append(resample_points(points, self.target_num_points))
        pointcloud_data = normalize_sequence_globally(np.asarray(frames, dtype=np.float32))

        metadata = {
            'sample_uid': lidar_ann['sample_uid'],
            'trial_name': lidar_ann['trial_name'],
            'subject': lidar_ann['subject'],
            'environment': lidar_ann['environment'],
            'activity': lidar_ann['activity'],
            'lighting_level': lidar_ann['lighting_level'],
        }
        return pose_data, torch.FloatTensor(pointcloud_data), int(lidar_ann['label']), metadata


def collate_fn(batch):
    pose_data_list = [item[0] for item in batch]
    pointcloud_data = torch.stack([item[1] for item in batch])
    labels = torch.LongTensor([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    return pose_data_list, pointcloud_data, labels, metadata
