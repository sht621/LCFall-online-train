#!/usr/bin/env python3
"""Save preview PNGs for ROI-only LiDAR samples or trials."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from online_train.lidar.preprocessing.pcd_io import load_xyz_pcd
from online_train.lidar.preprocessing.roi_frame_extractor import ManualROI, ROIFrameExtractor
from online_train.lidar.preprocessing.sample_normalization import normalize_sequence

DEFAULT_ROTATION = {"x": 1.1, "y": 27.8, "z": 0.0}
DEFAULT_ROI = {"x": [1.8, 7.5], "y": [-2.8, 3.0], "z": [-1.1, 1.0]}
MANIFEST_PATH = Path("/LCFall/src/online_train/manifests/all_samples_lidar.json")
ROI_CONFIG_PATH = Path("/LCFall/src/online_train/manifests/lidar_roi_config.json")
FRAME_ROOT = Path("/LCFall/src/online_train/generated/lidar_roi_frames")
DATASET_ROOT = Path("/LCFall/datasets/LCFall_dataset/data")
OUT_ROOT = Path("/LCFall/src/online_train/generated/lidar_visualization")




def load_trial_config(trial_name: str) -> dict:
    if ROI_CONFIG_PATH.exists():
        with open(ROI_CONFIG_PATH, "r") as f:
            config = json.load(f)
        return config.get("trials", {}).get(trial_name, config.get("default", {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}))
    return {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}

def load_manifest_sample(sample_uid: str) -> dict:
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    for sample in data.get("samples", []):
        if sample["sample_uid"] == sample_uid:
            return sample
    raise KeyError(sample_uid)


def load_points(stage: str, trial_name: str, frame_number: int) -> np.ndarray:
    config = load_trial_config(trial_name)
    rotation = config.get("rotation_degrees", DEFAULT_ROTATION)
    roi = config.get("roi", DEFAULT_ROI)
    if stage == "roi":
        return np.load(FRAME_ROOT / trial_name / f"{frame_number:05d}.npy").astype(np.float32)
    pcd_path = DATASET_ROOT / trial_name / "lidar" / f"{frame_number:05d}.pcd"
    raw_points = load_xyz_pcd(pcd_path)
    extractor = ROIFrameExtractor(rotation, ManualROI.from_dict(roi), target_num_points=256)
    if stage == "raw":
        return raw_points
    if stage == "rotated":
        return extractor.rotate(raw_points)
    raise ValueError(stage)


def save_frame_grid(frames: List[np.ndarray], title: str, output_path: Path) -> None:
    views = [(20, -135, "oblique"), (0, -90, "front"), (0, 0, "side"), (90, -90, "top")]
    fig = plt.figure(figsize=(16, 18))
    for row, frame_points in enumerate(frames):
        sampled = frame_points if len(frame_points) <= 4000 else frame_points[np.random.choice(len(frame_points), 4000, replace=False)]
        for col, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(len(frames), len(views), row * len(views) + col + 1, projection="3d")
            if len(sampled) > 0:
                ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=3.0, alpha=0.5)
            ax.set_title(f"frame {row + 1}: {view_name}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
    plt.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save LiDAR preview PNGs")
    parser.add_argument("--trial_name", type=str, default=None)
    parser.add_argument("--sample_uid", type=str, default=None)
    parser.add_argument("--stage", choices=["raw", "rotated", "roi", "normalized"], default="roi")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_uid:
        sample = load_manifest_sample(args.sample_uid)
        trial_name = sample["trial_name"]
        frame_numbers = sample["frame_numbers"]
    elif args.trial_name:
        trial_name = args.trial_name
        frame_numbers = [1, 24, 48, 72, 96]
    else:
        raise ValueError("Specify --trial_name or --sample_uid")

    indices = np.linspace(0, len(frame_numbers) - 1, num=min(5, len(frame_numbers)), dtype=int)
    selected = [frame_numbers[index] for index in indices]
    frames = [load_points("roi" if args.stage == "normalized" else args.stage, trial_name, frame_number) for frame_number in selected]
    if args.stage == "normalized":
        frames = [normalize_sequence(np.stack(frames, axis=0))[index] for index in range(len(frames))]
    output_path = Path(args.output) if args.output else OUT_ROOT / (args.sample_uid or args.trial_name) / f"preview_{args.stage}.png"
    save_frame_grid(frames, f"{args.stage}: {args.sample_uid or trial_name}", output_path)
    print(output_path)


if __name__ == "__main__":
    main()
