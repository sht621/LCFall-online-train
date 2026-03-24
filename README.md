# online_train

`/LCFall/src/online_train` is a dedicated offline retraining workspace for the online deployment setting.

Key rules in this workspace:
- Existing `v2` code is not edited.
- Reused files are copied in and tracked locally.
- Camera labels use `0 = non-falling`, `1 = falling`.
- LiDAR training data uses manual ROI only.
- Background subtraction, SOR, and RANSAC are intentionally disabled for training-data generation.
- Sample extraction is fixed to 48 frames and 3 samples per trial.

Typical pipeline:
```bash
python3 /LCFall/src/online_train/scripts/build_camera_pose_dataset.py
python3 /LCFall/src/online_train/scripts/build_lidar_roi_dataset.py
python3 /LCFall/src/online_train/scripts/build_sample_manifests.py
python3 /LCFall/src/online_train/scripts/build_dataset_splits.py
python3 /LCFall/src/online_train/scripts/build_camera_annotations.py
python3 /LCFall/src/online_train/scripts/train_all_online_retrain.py --steps camera_train lidar_train fusion_train metadata
```

## GitHub export

This workspace can be published without model weights. The repository ignore rules exclude:
- checkpoints and `*.pth`
- generated intermediate data
- annotation pickle files
- dataset-derived train/val/test manifests

Files that are safe and useful to publish include:
- source code under `camera/`, `lidar/`, `fusion_mlp/`, and `scripts/`
- lightweight configuration files such as `manifests/lidar_roi_config.json` and `manifests/split_config.json`
- design notes in `docs/`

Before publishing, review absolute paths like `/LCFall/...` and replace them with environment variables or project-relative paths if you want the repository to run outside this machine.
