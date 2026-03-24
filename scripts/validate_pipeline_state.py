#!/usr/bin/env python3
"""Validate current online_train pipeline state and print next actionable steps."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path('/LCFall/src/online_train')
CAMERA_POSE_ROOT = WORKSPACE / 'generated' / 'camera_pose'
LIDAR_ROOT = WORKSPACE / 'generated' / 'lidar_roi_frames'
MANIFEST_DIR = WORKSPACE / 'manifests'
ANNOTATION_DIR = WORKSPACE / 'annotations'
CAMERA_CKPT_DIR = WORKSPACE / 'camera' / 'checkpoints'
LIDAR_CKPT_DIR = WORKSPACE / 'lidar' / 'checkpoints'
FUSION_CKPT_DIR = WORKSPACE / 'fusion_mlp' / 'checkpoints'


def count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def json_sample_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, 'r') as f:
        data = json.load(f)
    return len(data.get('samples', []))


def pkl_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    state = {
        'camera_pose_json_count': count_files(CAMERA_POSE_ROOT, '*.json'),
        'lidar_roi_npy_count': count_files(LIDAR_ROOT, '*.npy'),
        'camera_manifest_samples_all': json_sample_count(MANIFEST_DIR / 'all_samples_camera.json'),
        'lidar_manifest_samples_all': json_sample_count(MANIFEST_DIR / 'all_samples_lidar.json'),
        'fusion_manifest_samples_all': json_sample_count(MANIFEST_DIR / 'all_samples_fusion.json'),
        'camera_manifest_samples_train': json_sample_count(MANIFEST_DIR / 'camera_train.json'),
        'camera_manifest_samples_val': json_sample_count(MANIFEST_DIR / 'camera_val.json'),
        'camera_manifest_samples_test': json_sample_count(MANIFEST_DIR / 'camera_test.json'),
        'lidar_manifest_samples_train': json_sample_count(MANIFEST_DIR / 'lidar_train.json'),
        'lidar_manifest_samples_val': json_sample_count(MANIFEST_DIR / 'lidar_val.json'),
        'lidar_manifest_samples_test': json_sample_count(MANIFEST_DIR / 'lidar_test.json'),
        'fusion_manifest_samples_train': json_sample_count(MANIFEST_DIR / 'fusion_train.json'),
        'fusion_manifest_samples_val': json_sample_count(MANIFEST_DIR / 'fusion_val.json'),
        'fusion_manifest_samples_test': json_sample_count(MANIFEST_DIR / 'fusion_test.json'),
        'camera_annotations_ready': all([
            pkl_exists(ANNOTATION_DIR / 'lcfall_online_retrain_train.pkl'),
            pkl_exists(ANNOTATION_DIR / 'lcfall_online_retrain_val.pkl'),
            pkl_exists(ANNOTATION_DIR / 'lcfall_online_retrain_test.pkl'),
        ]),
        'camera_best_ckpts': count_files(CAMERA_CKPT_DIR, 'best_f1_f1_score_epoch_*.pth'),
        'lidar_best_ckpts': count_files(LIDAR_CKPT_DIR, 'best_model.pth'),
        'fusion_best_ckpts': count_files(FUSION_CKPT_DIR, 'best_model.pth'),
    }

    next_steps = []
    if state['camera_pose_json_count'] == 0:
        next_steps.append('Run build_camera_pose_dataset.py after confirming the MMPose environment for RTMDet-M + ViTPose-S.')
    if state['lidar_roi_npy_count'] == 0:
        next_steps.append('Run build_lidar_roi_dataset.py to generate ROI-only LiDAR frames from LCFall_dataset lidar/*.pcd.')
    if min(state['camera_manifest_samples_all'], state['lidar_manifest_samples_all'], state['fusion_manifest_samples_all']) == 0:
        next_steps.append('Run build_sample_manifests.py after camera pose JSONs and LiDAR ROI frames are ready.')
    if min(
        state['camera_manifest_samples_train'],
        state['camera_manifest_samples_val'],
        state['camera_manifest_samples_test'],
        state['lidar_manifest_samples_train'],
        state['lidar_manifest_samples_val'],
        state['lidar_manifest_samples_test'],
        state['fusion_manifest_samples_train'],
        state['fusion_manifest_samples_val'],
        state['fusion_manifest_samples_test'],
    ) == 0:
        next_steps.append('Run build_dataset_splits.py to create subject-disjoint train/val/test manifests.')
    if not state['camera_annotations_ready']:
        next_steps.append('Run build_camera_annotations.py to generate MMAction PKL annotations from the sample manifests.')
    if state['camera_best_ckpts'] == 0 and state['camera_annotations_ready']:
        next_steps.append('Run camera training with the subject-split work_dir or train_all_online_retrain.py --steps camera_train.')
    if state['lidar_best_ckpts'] == 0 and state['lidar_manifest_samples_train'] > 0:
        next_steps.append('Run LiDAR training with train_lidar_online_retrain.py using lidar_train.json / lidar_val.json.')
    if state['fusion_best_ckpts'] == 0 and state['camera_best_ckpts'] > 0 and state['lidar_best_ckpts'] > 0:
        next_steps.append('Run fusion training with train_fusion_unified.py using fusion_train.json / fusion_val.json.')
    if not next_steps:
        next_steps.append('All tracked artifacts exist. Run export_training_metadata.py and spot-check saved visualizations.')

    payload = {
        'workspace': str(WORKSPACE),
        'state': state,
        'label_definition': {'0': 'non-falling', '1': 'falling'},
        'next_steps': next_steps,
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
