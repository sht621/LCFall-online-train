# Pipeline Design

## Camera
- Input images come from `/LCFall/datasets/LCFall_dataset/data` for `normal` and `/LCFall/datasets/LCFall_lowlight/<lighting>/data` for dark variants.
- `RTMDet-M + ViTPose-S` regenerates pose JSON files into `generated/camera_pose`.
- `build_camera_annotations.py` converts manifest-defined 48-frame windows into MMAction-compatible PKL files.

## LiDAR
- Input point clouds come from `/LCFall/datasets/LCFall_dataset/data/<trial>/lidar/*.pcd`.
- Processing is limited to rotation, manual ROI cropping, and 256-point resampling.
- Background subtraction, SOR, and RANSAC are not used in this training-data pipeline.
- Window-level normalization is applied at dataset loading time, not during frame generation.

## Training
- Camera, LiDAR, and fusion training keep the existing model structures whenever possible.
- Labels are unified to `0 = non-falling`, `1 = falling`.
- Falling F1 with `pos_label=1` is the primary selection metric.
