#!/usr/bin/env python3
"""Save MP4 videos for ROI-only LiDAR sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from online_train.lidar.preprocessing.sample_normalization import normalize_sequence
from online_train.lidar.visualization.save_lidar_preview import FRAME_ROOT, load_manifest_sample

OUT_ROOT = Path("/LCFall/src/online_train/generated/lidar_visualization")


def load_sequence(sample_uid: str, stage: str) -> tuple[dict, np.ndarray]:
    sample = load_manifest_sample(sample_uid)
    frames = [np.load(FRAME_ROOT / sample["trial_name"] / f"{frame_number:05d}.npy").astype(np.float32) for frame_number in sample["frame_numbers"]]
    sequence = np.stack(frames, axis=0)
    if stage == "normalized":
        sequence = normalize_sequence(sequence)
    return sample, sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save LiDAR MP4 previews")
    parser.add_argument("--sample_uid", required=True)
    parser.add_argument("--stage", choices=["roi", "normalized"], default="roi")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample, sequence = load_sequence(args.sample_uid, args.stage)
    output_path = Path(args.output) if args.output else OUT_ROOT / args.sample_uid / f"{args.stage}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(output_path), 150):
        for index, frame in enumerate(sequence, start=1):
            ax.clear()
            non_zero = np.linalg.norm(frame, axis=1) > 0
            points = frame[non_zero]
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.0, alpha=0.6)
            ax.set_xlim(np.min(sequence[..., 0]), np.max(sequence[..., 0]))
            ax.set_ylim(np.min(sequence[..., 1]), np.max(sequence[..., 1]))
            ax.set_zlim(np.min(sequence[..., 2]), np.max(sequence[..., 2]))
            ax.set_title(f"{args.sample_uid} | {args.stage} | frame {index:02d}/48")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=20, azim=-135)
            writer.grab_frame()
    plt.close(fig)
    print(json.dumps({"sample_uid": args.sample_uid, "stage": args.stage, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
