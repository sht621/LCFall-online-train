#!/usr/bin/env python3
"""Shared configuration for the online retraining workspace."""

from pathlib import Path

PROJECT_ROOT = Path('/LCFall')
EXPERIMENT_ROOT = PROJECT_ROOT / 'src/online_train'

UNIFIED_OPTIMIZER = {
    'type': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
    'clip_grad_max_norm': 1.0,
}

UNIFIED_LOSS = {
    'class_weight': [1.0, 5.0],
    'pos_label': 1,
    'label_definition': {'0': 'non-falling', '1': 'falling'},
}

UNIFIED_TRAINING = {
    'num_epochs': 100,
    'batch_size': 16,
    'dropout': 0.5,
    'num_workers': 0,
    'seed': 42,
}

UNIFIED_EARLY_STOPPING = {
    'monitor': 'f1_score',
    'patience': 15,
    'min_delta': 0.001,
    'mode': 'max',
}

OVERSAMPLING = {
    'normal': 5,
    'dark_10x': 1,
    'dark_50x': 1,
    'dark_100x': 1,
    'dark_200x': 1,
    'dark_300x': 1,
}

DATA_PATHS = {
    'camera_pose': str(EXPERIMENT_ROOT / 'generated/camera_pose'),
    'camera_annotations': str(EXPERIMENT_ROOT / 'annotations'),
    'lidar_raw': str(PROJECT_ROOT / 'datasets/LCFall_dataset/data'),
    'lidar_roi_frames': str(EXPERIMENT_ROOT / 'generated/lidar_roi_frames'),
    'sample_manifests': str(EXPERIMENT_ROOT / 'manifests'),
}

OUTPUT_DIRS = {
    'camera': {'checkpoints': str(EXPERIMENT_ROOT / 'camera/checkpoints')},
    'lidar': {'checkpoints': str(EXPERIMENT_ROOT / 'lidar/checkpoints')},
    'fusion_mlp': {'checkpoints': str(EXPERIMENT_ROOT / 'fusion_mlp/checkpoints')},
}

LOSO_FOLDS = {
    1: {'test': 'P01', 'val': 'P02', 'train': ['P03', 'P04', 'P05']},
    2: {'test': 'P02', 'val': 'P03', 'train': ['P01', 'P04', 'P05']},
    3: {'test': 'P03', 'val': 'P04', 'train': ['P01', 'P02', 'P05']},
    4: {'test': 'P04', 'val': 'P05', 'train': ['P01', 'P02', 'P03']},
    5: {'test': 'P05', 'val': 'P01', 'train': ['P02', 'P03', 'P04']},
}

EVALUATION = {
    'lighting_levels': ['normal', 'dark_10x', 'dark_50x', 'dark_100x', 'dark_200x', 'dark_300x'],
    'environment_mapping': {'E01': 'normal_env', 'E02': 'occluded', 'E03': 'distanced'},
}


def get_unified_config():
    return {
        'optimizer': UNIFIED_OPTIMIZER,
        'loss': UNIFIED_LOSS,
        'training': UNIFIED_TRAINING,
        'early_stopping': UNIFIED_EARLY_STOPPING,
        'oversampling': OVERSAMPLING,
        'data_paths': DATA_PATHS,
        'output_dirs': OUTPUT_DIRS,
        'loso_folds': LOSO_FOLDS,
        'evaluation': EVALUATION,
    }


if __name__ == '__main__':
    import json

    print(json.dumps(get_unified_config(), indent=2, default=str))
