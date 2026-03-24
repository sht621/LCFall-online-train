#!/usr/bin/env python3
"""Top-level runner for the online retraining offline pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path("/LCFall")
WORKSPACE_ROOT = PROJECT_ROOT / "src" / "online_train"
MMACTION_ROOT = PROJECT_ROOT / "external_libs" / "mmaction2"
LOG_DIR = WORKSPACE_ROOT / "generated" / "samples"


def run_command(cmd: List[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def build_camera_pose(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "build_camera_pose_dataset.py"),
        "--device",
        args.device,
    ]
    if args.trials:
        cmd.extend(["--trials", *args.trials])
    if args.overwrite:
        cmd.append("--overwrite")
    run_command(cmd)


def build_lidar_roi(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(WORKSPACE_ROOT / "scripts" / "build_lidar_roi_dataset.py")]
    if args.trials:
        cmd.extend(["--trials", *args.trials])
    if args.overwrite:
        cmd.append("--overwrite")
    run_command(cmd)


def build_manifests(_: argparse.Namespace) -> None:
    run_command([sys.executable, str(WORKSPACE_ROOT / "scripts" / "build_sample_manifests.py")])


def build_splits(_: argparse.Namespace) -> None:
    run_command([sys.executable, str(WORKSPACE_ROOT / "scripts" / "build_dataset_splits.py")])


def build_camera_annotations(_: argparse.Namespace) -> None:
    run_command([sys.executable, str(WORKSPACE_ROOT / "scripts" / "build_camera_annotations.py")])


def train_camera(args: argparse.Namespace) -> None:
    config_path = WORKSPACE_ROOT / "camera" / "configs" / "slowonly_r50_unified.py"
    work_dir = WORKSPACE_ROOT / "camera" / "checkpoints" / "train_subject_split"
    run_command([
        sys.executable,
        "tools/train.py",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--cfg-options",
        "env_cfg.cudnn_benchmark=False",
    ], cwd=MMACTION_ROOT)


def train_lidar(args: argparse.Namespace) -> None:
    run_command([
        sys.executable,
        str(WORKSPACE_ROOT / "lidar" / "training" / "train_lidar_online_retrain.py"),
    ])


def train_fusion(args: argparse.Namespace) -> None:
    run_command([
        sys.executable,
        str(WORKSPACE_ROOT / "fusion_mlp" / "train_fusion_unified.py"),
    ])


def export_metadata(_: argparse.Namespace) -> None:
    run_command([sys.executable, str(WORKSPACE_ROOT / "scripts" / "export_training_metadata.py")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the online_train offline retraining pipeline")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trials", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["camera_pose", "lidar_roi", "manifests", "splits", "camera_annotations", "camera_train", "lidar_train", "fusion_train", "metadata"],
        choices=["camera_pose", "lidar_roi", "manifests", "splits", "camera_annotations", "camera_train", "lidar_train", "fusion_train", "metadata"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = {
        "camera_pose": build_camera_pose,
        "lidar_roi": build_lidar_roi,
        "manifests": build_manifests,
        "splits": build_splits,
        "camera_annotations": build_camera_annotations,
        "camera_train": train_camera,
        "lidar_train": train_lidar,
        "fusion_train": train_fusion,
        "metadata": export_metadata,
    }
    for step in args.steps:
        print(f"\n=== {step} ===")
        steps[step](args)
    with open(LOG_DIR / "last_pipeline_run.json", "w") as f:
        json.dump({"steps": args.steps, "trials": args.trials or []}, f, indent=2)


if __name__ == "__main__":
    main()
