#!/usr/bin/env python3
"""Regenerate camera pose JSONs for online retraining.

Input:
- normal: /LCFall/datasets/LCFall_dataset/data/<trial>/rgb/*.png
- dark_*: /LCFall/datasets/LCFall_lowlight/<lighting>/data/<trial>/rgb/*.png

Output:
- /LCFall/src/online_train/generated/camera_pose/<lighting>/data/<trial>/rgb/*.json

The output format intentionally follows the existing skeleton JSON structure so the
annotation builder and MMAction training configs remain compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

LIGHTING_LEVELS = ["normal", "dark_10x", "dark_50x", "dark_100x", "dark_200x", "dark_300x"]
NORMAL_ROOT = Path("/LCFall/datasets/LCFall_dataset/data")
LOWLIGHT_ROOT = Path("/LCFall/datasets/LCFall_lowlight")
OUTPUT_ROOT = Path("/LCFall/src/online_train/generated/camera_pose")
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")


class PoseGenerator:
    def __init__(self, det_model: str, pose_model: str, det_cat_ids: int = 0, device: str = "cuda:0", det_weights: str | None = None, pose_weights: str | None = None):
        self.det_model = det_model
        self.pose_model = pose_model
        self.det_cat_ids = det_cat_ids
        self.device = device
        self.det_weights = det_weights
        self.pose_weights = pose_weights
        self._inferencer = None

    def _load_inferencer(self):
        if self._inferencer is not None:
            return self._inferencer
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise RuntimeError(
                "MMPoseInferencer is not available. Install mmpose/mmengine/mmcv before running pose regeneration."
            ) from exc
        kwargs = dict(
            pose2d=self.pose_model,
            det_model=self.det_model,
            det_cat_ids=self.det_cat_ids,
            device=self.device,
        )
        if self.pose_weights:
            kwargs["pose2d_weights"] = self.pose_weights
        if self.det_weights:
            kwargs["det_weights"] = self.det_weights
        self._inferencer = MMPoseInferencer(**kwargs)
        return self._inferencer

    def infer_one(self, image_path: Path) -> Dict:
        inferencer = self._load_inferencer()
        result_iter = inferencer(str(image_path), return_vis=False, draw_bbox=False, show=False)
        result = next(result_iter)
        predictions = result.get("predictions", [])
        persons = predictions[0] if predictions else []
        return self._convert_result(image_path, persons)

    def infer_many(self, image_paths: List[Path], batch_size: int = 16) -> Iterator[Dict]:
        if not image_paths:
            return
        inferencer = self._load_inferencer()
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start:start + batch_size]
            result_iter = inferencer(
                [str(path) for path in chunk],
                return_vis=False,
                draw_bbox=False,
                show=False,
            )
            for image_path in chunk:
                result = next(result_iter)
                predictions = result.get("predictions", [])
                persons = predictions[0] if predictions else []
                yield self._convert_result(image_path, persons)

    def _convert_result(self, image_path: Path, persons: List[Dict]) -> Dict:
        frame_index = int(image_path.stem) - 1 if image_path.stem.isdigit() else 0
        if not persons:
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "frame_index": frame_index,
                "num_persons": 0,
                "detection_status": "no_person",
                "detection_threshold": 0.1,
                "model_id": {"det": self.det_model, "pose": self.pose_model},
                "persons": [],
            }

        def score_of(person: Dict) -> float:
            if "bbox_score" in person:
                score = person["bbox_score"]
                if isinstance(score, list):
                    return float(score[0]) if score else 0.0
                return float(score)
            return float(person.get("score", 0.0))

        best = max(persons, key=score_of)
        keypoints = best.get("keypoints", [])
        scores = best.get("keypoint_scores", best.get("keypoints_visible", []))
        bbox = best.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox_score = best.get("bbox_score", [score_of(best)])
        if bbox and isinstance(bbox[0], list):
            bbox = bbox[0]
        if not scores:
            scores = [0.0] * len(keypoints)
        while len(keypoints) < 17:
            keypoints.append([0.0, 0.0])
        while len(scores) < 17:
            scores.append(0.0)
        scores = [float(s) for s in scores[:17]]
        coordinates = [[float(x), float(y)] for x, y in keypoints[:17]]
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "frame_index": frame_index,
            "num_persons": 1,
            "detection_status": "success",
            "detection_threshold": 0.1,
            "model_id": {"det": self.det_model, "pose": self.pose_model, "det_weights": self.det_weights, "pose_weights": self.pose_weights},
            "persons": [
                {
                    "person_id": 0,
                    "det_score": float(score_of(best)),
                    "bbox": [float(v) for v in bbox[:4]],
                    "bbox_score": [float(v) for v in (bbox_score if isinstance(bbox_score, list) else [bbox_score])],
                    "keypoints": {
                        "coordinates": coordinates,
                        "scores": scores,
                    },
                    "kpt_score_stats": {
                        "mean": float(sum(scores) / len(scores)) if scores else 0.0,
                        "min": float(min(scores)) if scores else 0.0,
                        "max": float(max(scores)) if scores else 0.0,
                    },
                }
            ],
        }


def resolve_input_root(lighting: str) -> Path:
    return NORMAL_ROOT if lighting == "normal" else LOWLIGHT_ROOT / lighting / "data"


def iter_trial_dirs(root: Path, trial_names: Optional[set[str]] = None) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("P"):
            continue
        if trial_names and path.name not in trial_names:
            continue
        yield path


def collect_image_paths(rgb_dir: Path) -> List[Path]:
    images: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(rgb_dir.glob(pattern))
    return sorted(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RTMDet-M + ViTPose-S pose JSONs for online_train")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--det_model", type=str, default="/usr/local/lib/python3.10/dist-packages/mmpose/.mim/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py")
    parser.add_argument("--det_weights", type=str, default="/LCFall/weights/camera/mmdetection/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
    parser.add_argument("--pose_model", type=str, default="vitpose-s")
    parser.add_argument("--pose_weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lighting", nargs="*", default=LIGHTING_LEVELS)
    parser.add_argument("--trials", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trial_names = set(args.trials) if args.trials else None
    generator = PoseGenerator(args.det_model, args.pose_model, device=args.device, det_weights=args.det_weights, pose_weights=args.pose_weights)

    summary = {"label_definition": {"0": "non-falling", "1": "falling"}, "lighting": {}}
    for lighting in args.lighting:
        input_root = resolve_input_root(lighting)
        if not input_root.exists():
            print(f"[skip] missing input root: {input_root}")
            continue
        lighting_out = output_root / lighting / "data"
        lighting_out.mkdir(parents=True, exist_ok=True)
        trial_count = 0
        frame_count = 0
        for trial_dir in iter_trial_dirs(input_root, trial_names):
            rgb_dir = trial_dir / "rgb"
            if not rgb_dir.exists():
                continue
            image_paths = collect_image_paths(rgb_dir)
            if not image_paths:
                continue
            out_dir = lighting_out / trial_dir.name / "rgb"
            out_dir.mkdir(parents=True, exist_ok=True)
            pending_paths: List[Path] = []
            for image_path in image_paths:
                out_path = out_dir / f"{image_path.stem}.json"
                if out_path.exists() and not args.overwrite:
                    frame_count += 1
                    continue
                pending_paths.append(image_path)
            for payload in generator.infer_many(pending_paths, batch_size=args.batch_size):
                out_path = out_dir / f"{Path(payload['image_name']).stem}.json"
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                frame_count += 1
            trial_count += 1
        summary["lighting"][lighting] = {"input_root": str(input_root), "trials": trial_count, "frames": frame_count}

    with open(output_root / "pose_generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
