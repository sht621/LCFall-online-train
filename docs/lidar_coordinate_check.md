# LiDAR Coordinate Check

Use `lidar/preprocessing/verify_coordinate_system.py` to save four-view PNGs for:
- raw point cloud
- rotated point cloud
- ROI-only point cloud
- resampled 256-point output

Checkpoints for acceptance:
- floor looks approximately horizontal
- height direction is stable across frames
- standing samples are tall and upright
- falling samples show reduced height and wider spread

If a correction is needed, update only:
- rotation angles
- axis sign convention
- ROI values

Record the before/after images and the reason in this file.
