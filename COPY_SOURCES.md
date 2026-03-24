# Copy Sources

| copy_from | copy_to | base_reason | planned_change |
|---|---|---|---|
| `/LCFall/src/unified_comparison/unified_config.py` | `/LCFall/src/online_train/unified_config.py` | unified settings baseline | workspace paths, label definition, class weights |
| `/LCFall/src/unified_comparison/fusion_mlp/train_fusion_unified.py` | `/LCFall/src/online_train/fusion_mlp/train_fusion_unified.py` | fusion training baseline | sample-manifest input, new labels, new paths |
| `/LCFall/src/unified_comparison/scripts/train_all_unified.py` | `/LCFall/src/online_train/scripts/train_all_online_retrain.py` | top-level runner baseline | rebuild around offline retraining pipeline |
| `/LCFall/src/camera/fair_comparison/lowlight_dataset.py` | `/LCFall/src/online_train/camera/support/lowlight_dataset.py` | camera dataset support | imports only |
| `/LCFall/src/camera/fair_comparison/f1_evaluator.py` | `/LCFall/src/online_train/camera/support/f1_evaluator.py` | evaluator baseline | `pos_label=1`, new label definition |
| `/LCFall/src/unified_comparison/lcfall_only_experiment/scripts/create_lcfall_only_annotations.py` | `/LCFall/src/online_train/scripts/build_camera_annotations.py` | LCFall-only annotation baseline | sample-level 48-frame annotations, new labels |
| `/LCFall/src/unified_comparison/lcfall_only_experiment/lidar/train_lidar_lcfall_only.py` | `/LCFall/src/online_train/lidar/training/train_lidar_online_retrain.py` | LCFall-only LiDAR training baseline | manifest-based loader, new labels |
| `/LCFall/src/lidar/preprocessing/batch_preprocess_lcfall.py` | `/LCFall/src/online_train/scripts/build_lidar_roi_dataset.py` | LiDAR preprocessing baseline | ROI-only, no background subtraction, no SOR, no RANSAC |
| `/LCFall/src/lidar/preprocessing/verify_coordinate_system.py` | `/LCFall/src/online_train/lidar/preprocessing/verify_coordinate_system.py` | coordinate-check baseline | save-only ROI verification for online_train |
| `/LCFall/src/lidar/preprocessing/visualize_rotation_and_roi.py` | `/LCFall/src/online_train/lidar/visualization/save_lidar_preview.py` | LiDAR visualization baseline | file-save preview for raw/rotated/roi/normalized stages |
