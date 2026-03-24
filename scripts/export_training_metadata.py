#!/usr/bin/env python3
"""Export a compact metadata snapshot for the online_train workspace."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE_ROOT = Path("/LCFall/src/online_train")
OUTPUT_PATH = WORKSPACE_ROOT / "generated" / "samples" / "training_metadata.json"


def maybe_load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    return {"path": str(path), "exists": True}


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workspace_root": str(WORKSPACE_ROOT),
        "label_definition": {"0": "non-falling", "1": "falling"},
        "camera_annotations": maybe_load(WORKSPACE_ROOT / "annotations" / "annotation_summary.json"),
        "camera_pose_summary": maybe_load(WORKSPACE_ROOT / "generated" / "camera_pose" / "pose_generation_summary.json"),
        "lidar_roi_summary": maybe_load(WORKSPACE_ROOT / "generated" / "lidar_roi_frames" / "roi_build_summary.json"),
        "sample_manifest": maybe_load(WORKSPACE_ROOT / "manifests" / "all_samples_fusion.json"),
        "camera_checkpoints": sorted(str(p) for p in (WORKSPACE_ROOT / "camera" / "checkpoints").glob("*/best_f1_f1_score_epoch_*.pth")),
        "lidar_checkpoints": sorted(str(p) for p in (WORKSPACE_ROOT / "lidar" / "checkpoints").glob("*/best_model.pth")),
        "fusion_checkpoints": sorted(str(p) for p in (WORKSPACE_ROOT / "fusion_mlp" / "checkpoints").glob("*/best_model.pth")),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
