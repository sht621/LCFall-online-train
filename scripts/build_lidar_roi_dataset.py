#!/usr/bin/env python3
"""Build ROI-only LiDAR frame dataset for online retraining.

Input:
- /LCFall/datasets/LCFall_dataset/data/<trial>/lidar/*.pcd

Output:
- /LCFall/src/online_train/generated/lidar_roi_frames/<trial>/*.npy

This script intentionally does not perform background subtraction, SOR, or RANSAC.
It keeps only coordinate rotation, manual ROI cropping, and 256-point resampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from typing import Dict, Iterable, List

import numpy as np

from online_train.lidar.preprocessing.pcd_io import load_xyz_pcd
from online_train.lidar.preprocessing.roi_frame_extractor import ManualROI, ROIFrameExtractor

INPUT_ROOT = Path("/LCFall/datasets/LCFall_dataset/data")
OUTPUT_ROOT = Path("/LCFall/src/online_train/generated/lidar_roi_frames")
DEFAULT_ROTATION = {"x": 1.1, "y": 27.8, "z": 0.0}
DEFAULT_ROI = {"x": [1.8, 7.5], "y": [-2.8, 3.0], "z": [-1.1, 1.0]}
ROI_CONFIG_PATH = Path("/LCFall/src/online_train/manifests/lidar_roi_config.json")


def load_roi_config() -> Dict:
    if ROI_CONFIG_PATH.exists():
        with open(ROI_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"default": {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}}


def resolve_trial_config(config: Dict, trial_name: str) -> Dict:
    return config.get("trials", {}).get(trial_name, config.get("default", {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}))


def iter_trials(input_root: Path, trials: set[str] | None) -> Iterable[Path]:
    for path in sorted(input_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("P"):
            continue
        if trials and path.name not in trials:
            continue
        if (path / "lidar").exists():
            yield path


def load_pcd_points(pcd_path: Path) -> np.ndarray:
    return load_xyz_pcd(pcd_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ROI-only LiDAR frames for online_train")
    parser.add_argument("--input_root", type=str, default=str(INPUT_ROOT))
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--trials", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trials = set(args.trials) if args.trials else None
    roi_config = load_roi_config()

    summary = {"input_root": str(input_root), "output_root": str(output_root), "trials": {}, "notes": ["background subtraction disabled", "SOR disabled", "RANSAC disabled"]}
    for trial_dir in iter_trials(input_root, trials):
        config = resolve_trial_config(roi_config, trial_dir.name)
        extractor = ROIFrameExtractor(
            rotation_degrees=config.get("rotation_degrees", DEFAULT_ROTATION),
            roi=ManualROI.from_dict(config.get("roi", DEFAULT_ROI)),
            target_num_points=256,
        )
        lidar_dir = trial_dir / "lidar"
        frame_paths = sorted(lidar_dir.glob("*.pcd"))
        trial_out = output_root / trial_dir.name
        trial_out.mkdir(parents=True, exist_ok=True)
        point_counts: List[int] = []
        for pcd_path in frame_paths:
            out_path = trial_out / f"{pcd_path.stem}.npy"
            if out_path.exists() and not args.overwrite:
                existing = np.load(out_path)
                point_counts.append(int(np.count_nonzero(np.linalg.norm(existing, axis=1))))
                continue
            raw_points = load_pcd_points(pcd_path)
            processed = extractor.process_points(raw_points)
            np.save(out_path, processed.astype(np.float32))
            point_counts.append(int(np.count_nonzero(np.linalg.norm(processed, axis=1))))
        summary["trials"][trial_dir.name] = {
            "frames": len(frame_paths),
            "output_dir": str(trial_out),
            "rotation_degrees": config.get("rotation_degrees", DEFAULT_ROTATION),
            "roi": config.get("roi", DEFAULT_ROI),
            "mean_non_zero_points": float(np.mean(point_counts)) if point_counts else 0.0,
        }

    with open(output_root / "roi_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not ROI_CONFIG_PATH.exists():
        with open(ROI_CONFIG_PATH, "w") as f:
            json.dump({"default": {"rotation_degrees": DEFAULT_ROTATION, "roi": DEFAULT_ROI}, "trials": {}}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
