#!/usr/bin/env python3
"""Build sample manifests for camera, LiDAR, and fusion training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LIGHTING_LEVELS = ['normal', 'dark_10x', 'dark_50x', 'dark_100x', 'dark_200x', 'dark_300x']
FALL_ACTIONS = {'bf', 'cf', 'ff', 'sf'}
NON_FALL_ACTIONS = {'wk', 'st', 'sq', 'tr', 'ly', 'pu'}
WINDOW_SIZE = 48

CAMERA_POSE_ROOT = Path('/LCFall/src/online_train/generated/camera_pose')
LIDAR_ROOT = Path('/LCFall/src/online_train/generated/lidar_roi_frames')
MANIFEST_DIR = Path('/LCFall/src/online_train/manifests')


def parse_trial_name(trial_name: str) -> Dict[str, str]:
    subject, environment, activity, trial = trial_name.split('_')
    return {'subject': subject, 'environment': environment, 'activity': activity, 'trial': trial}


def resolve_label(activity: str) -> int:
    if activity in FALL_ACTIONS:
        return 1
    if activity in NON_FALL_ACTIONS:
        return 0
    raise ValueError(f'Unknown activity: {activity}')


def clamp_start(start: int, length: int) -> int:
    max_start = max(length - WINDOW_SIZE, 0)
    return max(0, min(start, max_start))


def make_non_fall_windows(length: int) -> List[int]:
    max_start = max(length - WINDOW_SIZE, 0)
    return [0, max_start // 2, max_start]


def make_fall_windows(length: int) -> List[int]:
    center = length // 2
    return [
        clamp_start(center - 32, length),
        clamp_start(center - 24, length),
        clamp_start(center - 16, length),
    ]


def frame_windows(frame_count: int, label: int) -> List[int]:
    return make_non_fall_windows(frame_count) if label == 0 else make_fall_windows(frame_count)


def build_sample_record(lighting: str, trial_name: str, frame_count: int, start: int, index: int, meta: Dict[str, str], label: int) -> Dict:
    frame_numbers = list(range(start + 1, start + WINDOW_SIZE + 1))
    sample_uid = f'{lighting}__{trial_name}__S{index}'
    return {
        'sample_uid': sample_uid,
        'source_trial': trial_name,
        'trial_name': trial_name,
        'subject': meta['subject'],
        'environment': meta['environment'],
        'activity': meta['activity'],
        'trial': meta['trial'],
        'lighting_level': lighting,
        'label': label,
        'total_frames': frame_count,
        'start_frame': start + 1,
        'end_frame': start + WINDOW_SIZE,
        'frame_numbers': frame_numbers,
    }


def iter_camera_trials() -> Iterable[Tuple[str, Path]]:
    for lighting in LIGHTING_LEVELS:
        lighting_root = CAMERA_POSE_ROOT / lighting / 'data'
        if not lighting_root.exists():
            continue
        for trial_dir in sorted(path for path in lighting_root.iterdir() if path.is_dir()):
            yield lighting, trial_dir


def build_camera_samples() -> List[Dict]:
    samples = []
    for lighting, trial_dir in iter_camera_trials():
        json_files = sorted((trial_dir / 'rgb').glob('*.json'))
        frame_count = len(json_files)
        if frame_count < WINDOW_SIZE:
            continue
        meta = parse_trial_name(trial_dir.name)
        label = resolve_label(meta['activity'])
        for index, start in enumerate(frame_windows(frame_count, label), start=1):
            samples.append(build_sample_record(lighting, trial_dir.name, frame_count, start, index, meta, label))
    return samples


def build_lidar_samples() -> List[Dict]:
    samples = []
    if not LIDAR_ROOT.exists():
        return samples
    for trial_dir in sorted(path for path in LIDAR_ROOT.iterdir() if path.is_dir()):
        npy_files = sorted(trial_dir.glob('*.npy'))
        frame_count = len(npy_files)
        if frame_count < WINDOW_SIZE:
            continue
        meta = parse_trial_name(trial_dir.name)
        label = resolve_label(meta['activity'])
        for lighting in LIGHTING_LEVELS:
            for index, start in enumerate(frame_windows(frame_count, label), start=1):
                samples.append(build_sample_record(lighting, trial_dir.name, frame_count, start, index, meta, label))
    return samples


def build_fusion_samples() -> List[Dict]:
    samples = []
    if not LIDAR_ROOT.exists():
        return samples
    for lighting, trial_dir in iter_camera_trials():
        json_files = sorted((trial_dir / 'rgb').glob('*.json'))
        lidar_dir = LIDAR_ROOT / trial_dir.name
        lidar_files = sorted(lidar_dir.glob('*.npy'))
        frame_count = min(len(json_files), len(lidar_files))
        if frame_count < WINDOW_SIZE:
            continue
        meta = parse_trial_name(trial_dir.name)
        label = resolve_label(meta['activity'])
        for index, start in enumerate(frame_windows(frame_count, label), start=1):
            samples.append(build_sample_record(lighting, trial_dir.name, frame_count, start, index, meta, label))
    return samples


def write_manifest(name: str, samples: List[Dict]) -> None:
    payload = {
        'label_definition': {'0': 'non-falling', '1': 'falling'},
        'window_size': WINDOW_SIZE,
        'sampling_rule': {
            'non_falling': 'start=[0, (N-48)//2, N-48]',
            'falling': 'c=N//2, start=[clip(c-32), clip(c-24), clip(c-16)]',
        },
        'samples': samples,
    }
    with open(MANIFEST_DIR / name, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    camera_samples = build_camera_samples()
    lidar_samples = build_lidar_samples()
    fusion_samples = build_fusion_samples()
    write_manifest('all_samples_camera.json', camera_samples)
    write_manifest('all_samples_lidar.json', lidar_samples)
    write_manifest('all_samples_fusion.json', fusion_samples)
    print(json.dumps({
        'camera_samples': len(camera_samples),
        'lidar_samples': len(lidar_samples),
        'fusion_samples': len(fusion_samples),
    }, indent=2))


if __name__ == '__main__':
    main()
