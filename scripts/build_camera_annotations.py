#!/usr/bin/env python3
"""Build sample-level camera annotations for the online retraining workspace."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TRAIN_MANIFEST_PATH = Path("/LCFall/src/online_train/manifests/camera_train.json")
VAL_MANIFEST_PATH = Path("/LCFall/src/online_train/manifests/camera_val.json")
TEST_MANIFEST_PATH = Path("/LCFall/src/online_train/manifests/camera_test.json")
POSE_ROOT = Path("/LCFall/src/online_train/generated/camera_pose")
OUTPUT_DIR = Path("/LCFall/src/online_train/annotations")
OVERSAMPLING_CONFIG = {"normal": 5, "dark_10x": 1, "dark_50x": 1, "dark_100x": 1, "dark_200x": 1, "dark_300x": 1}


def parse_sequence_name(seq_name: str) -> Optional[Dict[str, str]]:
    match = re.match(r"P(\d+)_E(\d+)_(.+)_T(\d+)", seq_name)
    if not match:
        return None
    return {
        "subject": f"P{int(match.group(1)):02d}",
        "environment": f"E{int(match.group(2)):02d}",
        "activity": match.group(3),
        "trial": f"T{int(match.group(4)):02d}",
    }


def load_pose_window(lighting: str, trial_name: str, frame_numbers: List[int]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rgb_dir = POSE_ROOT / lighting / "data" / trial_name / "rgb"
    keypoints: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    det_scores: List[float] = []
    no_detection = 0

    for frame_number in frame_numbers:
        json_path = rgb_dir / f"{frame_number:05d}.json"
        if not json_path.exists():
            keypoints.append(np.zeros((17, 2), dtype=np.float32))
            scores.append(np.zeros((17,), dtype=np.float32))
            no_detection += 1
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        persons = data.get("persons", [])
        if not persons:
            keypoints.append(np.zeros((17, 2), dtype=np.float32))
            scores.append(np.zeros((17,), dtype=np.float32))
            no_detection += 1
            continue
        person = persons[0]
        kp = person.get("keypoints", {})
        coords = np.asarray(kp.get("coordinates", np.zeros((17, 2))), dtype=np.float32)
        kps = np.asarray(kp.get("scores", np.zeros((17,))), dtype=np.float32)
        if coords.shape != (17, 2):
            coords_fixed = np.zeros((17, 2), dtype=np.float32)
            coords_fixed[: min(len(coords), 17)] = coords[:17]
            coords = coords_fixed
        if kps.shape != (17,):
            score_fixed = np.zeros((17,), dtype=np.float32)
            score_fixed[: min(len(kps), 17)] = kps[:17]
            kps = score_fixed
        keypoints.append(coords)
        scores.append(kps)
        det_scores.append(float(person.get("det_score", 0.0)))

    kp_array = np.asarray(keypoints, dtype=np.float32)[np.newaxis, ...]
    score_array = np.asarray(scores, dtype=np.float32)[np.newaxis, ...]
    stats = {
        "total_frames": len(frame_numbers),
        "no_detection_frames": no_detection,
        "detection_rate": (len(frame_numbers) - no_detection) / len(frame_numbers) if frame_numbers else 0.0,
        "mean_det_score": float(np.mean(det_scores)) if det_scores else 0.0,
        "mean_kpt_score": float(np.mean(score_array)) if score_array.size else 0.0,
    }
    return kp_array, score_array, stats


def build_annotation(sample: Dict, duplicate_index: int = 0) -> Dict:
    parsed = parse_sequence_name(sample["trial_name"])
    if parsed is None:
        raise ValueError(f"Failed to parse trial name: {sample['trial_name']}")
    keypoints, keypoint_scores, stats = load_pose_window(sample["lighting_level"], sample["trial_name"], sample["frame_numbers"])
    return {
        "frame_dir": f"{sample['lighting_level']}/data/{sample['trial_name']}",
        "label": int(sample["label"]),
        "img_shape": (480, 640),
        "original_shape": (480, 640),
        "total_frames": 48,
        "keypoint": keypoints,
        "keypoint_score": keypoint_scores,
        "sample_id": sample["sample_uid"],
        "sample_uid": sample["sample_uid"],
        "duplicate_index": duplicate_index,
        "subject": parsed["subject"],
        "environment": parsed["environment"],
        "activity": parsed["activity"],
        "trial": parsed["trial"],
        "lighting_level": sample["lighting_level"],
        "dataset": "lcfall",
        "frame_numbers": sample["frame_numbers"],
        "detection_stats": stats,
        "label_definition": {"0": "non-falling", "1": "falling"},
    }


def build_all_annotations(samples: List[Dict], oversample: bool) -> List[Dict]:
    annotations: List[Dict] = []
    for sample in samples:
        repeat = OVERSAMPLING_CONFIG.get(sample["lighting_level"], 1) if oversample else 1
        for duplicate_index in range(repeat):
            annotations.append(build_annotation(sample, duplicate_index=duplicate_index))
    return annotations


def subject_filter(annotations: List[Dict], subjects: List[str], train_mode: bool = False) -> List[Dict]:
    output: List[Dict] = []
    subjects_set = set(subjects)
    for ann in annotations:
        if ann["subject"] not in subjects_set:
            continue
        if not train_mode and ann["duplicate_index"] != 0:
            continue
        output.append(ann)
    return output


def write_pickle(path: Path, payload: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_MANIFEST_PATH, "r") as f:
        train_manifest = json.load(f)
    with open(VAL_MANIFEST_PATH, "r") as f:
        val_manifest = json.load(f)
    with open(TEST_MANIFEST_PATH, "r") as f:
        test_manifest = json.load(f)

    train_annotations = build_all_annotations(train_manifest.get("samples", []), oversample=True)
    val_annotations = build_all_annotations(val_manifest.get("samples", []), oversample=False)
    test_annotations = build_all_annotations(test_manifest.get("samples", []), oversample=False)
    all_annotations = train_annotations + val_annotations + test_annotations

    write_pickle(OUTPUT_DIR / "lcfall_online_retrain.pkl", all_annotations)
    write_pickle(OUTPUT_DIR / "lcfall_online_retrain_train.pkl", train_annotations)
    write_pickle(OUTPUT_DIR / "lcfall_online_retrain_val.pkl", val_annotations)
    write_pickle(OUTPUT_DIR / "lcfall_online_retrain_test.pkl", test_annotations)

    summary: Dict[str, object] = {
        "label_definition": {"0": "non-falling", "1": "falling"},
        "all_annotations": len(all_annotations),
        "train": len(train_annotations),
        "val": len(val_annotations),
        "test": len(test_annotations),
        "train_subjects": train_manifest.get("train_subjects", []),
        "val_subjects": val_manifest.get("val_subjects", []),
        "test_subjects": test_manifest.get("test_subjects", []),
    }

    with open(OUTPUT_DIR / "annotation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
