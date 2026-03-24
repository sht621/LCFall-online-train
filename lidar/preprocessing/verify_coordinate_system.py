#!/usr/bin/env python3
"""Save ROI-only LiDAR coordinate verification images for online_train."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from online_train.lidar.preprocessing.pcd_io import load_xyz_pcd
from online_train.lidar.preprocessing.roi_frame_extractor import ManualROI, ROIFrameExtractor

DEFAULT_ROTATION = {"x": 1.1, "y": 27.8, "z": 0.0}
DEFAULT_ROI = {"x": [1.8, 7.5], "y": [-2.8, 3.0], "z": [-1.1, 1.0]}
ROI_CONFIG_PATH = Path("/LCFall/src/online_train/manifests/lidar_roi_config.json")


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".pcd":
        return load_xyz_pcd(path)
    return np.load(path).astype(np.float32)




def load_trial_config(trial_name: str) -> dict:
    if ROI_CONFIG_PATH.exists():
        with open(ROI_CONFIG_PATH, "r") as f:
            config = json.load(f)
        return config.get("trials", {}).get(trial_name, config.get("default", {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}))
    return {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}

def sample_points(points: np.ndarray, limit: int = 5000) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.random.choice(len(points), limit, replace=False)
    return points[indices]


def draw_views(points: np.ndarray, title_prefix: str, output_path: Path, roi: dict | None = None) -> None:
    fig = plt.figure(figsize=(14, 10))
    views = [(20, -135, "oblique"), (0, -90, "front"), (0, 0, "side"), (90, -90, "top")]
    sampled = sample_points(points)
    for index, (elev, azim, name) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        if len(sampled) > 0:
            ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=2.0, alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title_prefix}\n{name}")
        ax.view_init(elev=elev, azim=azim)
        if roi is not None:
            ax.text2D(0.02, 0.02, json.dumps(roi), transform=ax.transAxes, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify coordinate system using raw and ROI-only LiDAR frames")
    parser.add_argument("--trial_dir", type=str, default="/LCFall/datasets/LCFall_dataset/data/P01_E01_bf_T01")
    parser.add_argument("--frame", type=str, default="00050")
    parser.add_argument("--output_dir", type=str, default="/LCFall/src/online_train/generated/lidar_visualization/coordinate_check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_dir = Path(args.trial_dir)
    pcd_path = trial_dir / "lidar" / f"{args.frame}.pcd"
    raw_points = load_points(pcd_path)
    config = load_trial_config(trial_dir.name)
    rotation = config.get("rotation_degrees", DEFAULT_ROTATION)
    roi = config.get("roi", DEFAULT_ROI)
    extractor = ROIFrameExtractor(rotation, ManualROI.from_dict(roi), target_num_points=256)
    rotated = extractor.rotate(raw_points)
    cropped = extractor.crop(rotated)
    resampled = extractor.resample(cropped)
    output_dir = Path(args.output_dir)
    draw_views(raw_points, f"raw: {trial_dir.name}/{args.frame}", output_dir / f"{trial_dir.name}_{args.frame}_raw.png")
    draw_views(rotated, f"rotated: {trial_dir.name}/{args.frame}", output_dir / f"{trial_dir.name}_{args.frame}_rotated.png", roi=roi)
    draw_views(cropped, f"roi-only: {trial_dir.name}/{args.frame}", output_dir / f"{trial_dir.name}_{args.frame}_roi.png", roi=roi)
    draw_views(resampled, f"resampled-256: {trial_dir.name}/{args.frame}", output_dir / f"{trial_dir.name}_{args.frame}_resampled.png", roi=roi)
    summary = {
        "trial_dir": str(trial_dir),
        "frame": args.frame,
        "raw_points": int(len(raw_points)),
        "roi_points": int(len(cropped)),
        "resampled_points": int(len(resampled)),
        "rotation_degrees": rotation,
        "roi": roi,
        "label_definition": {"0": "non-falling", "1": "falling"},
    }
    with open(output_dir / f"{trial_dir.name}_{args.frame}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
