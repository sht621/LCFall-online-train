#!/usr/bin/env python3
"""
F1Scoreを含むカスタム評価メトリクス

online_train のラベル定義:
- 0 = non-falling
- 1 = falling
"""

from typing import Dict, List, Optional, Sequence
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS


@METRICS.register_module()
class F1Metric(BaseMetric):
    """Compute binary metrics with falling(1) as the positive class."""

    default_prefix: Optional[str] = 'f1'

    def __init__(self, pos_label: int = 1, collect_device: str = 'cpu', prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pos_label = pos_label

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_scores = data_sample.get('pred_score')
            gt_label = data_sample['gt_label']
            pred_label = pred_scores.argmax(dim=0) if pred_scores is not None else data_sample['pred_label']
            self.results.append({
                'pred_label': pred_label.cpu() if isinstance(pred_label, torch.Tensor) else pred_label,
                'gt_label': gt_label.cpu() if isinstance(gt_label, torch.Tensor) else gt_label,
                'pred_score': pred_scores.cpu() if isinstance(pred_scores, torch.Tensor) else pred_scores,
            })

    def compute_metrics(self, results: List) -> Dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred = result['pred_label']
            gt = result['gt_label']
            if isinstance(pred, torch.Tensor):
                pred = pred.numpy()
            if isinstance(gt, torch.Tensor):
                gt = gt.numpy()
            pred_labels.append(int(pred.item()) if hasattr(pred, 'item') else int(pred))
            gt_labels.append(int(gt.item()) if hasattr(gt, 'item') else int(gt))

        pred_labels = np.asarray(pred_labels)
        gt_labels = np.asarray(gt_labels)
        tp = np.sum((gt_labels == self.pos_label) & (pred_labels == self.pos_label))
        fp = np.sum((gt_labels != self.pos_label) & (pred_labels == self.pos_label))
        tn = np.sum((gt_labels != self.pos_label) & (pred_labels != self.pos_label))
        fn = np.sum((gt_labels == self.pos_label) & (pred_labels != self.pos_label))

        total = len(gt_labels)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top1': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        }


@METRICS.register_module()
class F1MetricWithDump(F1Metric):
    def __init__(self, pos_label: int = 1, out_file_path: Optional[str] = None, collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(pos_label=pos_label, collect_device=collect_device, prefix=prefix)
        self.out_file_path = out_file_path

    def compute_metrics(self, results: List) -> Dict:
        metrics = super().compute_metrics(results)
        if self.out_file_path is not None:
            out_path = Path(self.out_file_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'wb') as f:
                pickle.dump({'metrics': metrics, 'predictions': results}, f)
        return metrics


@METRICS.register_module()
class F1MetricWithJSONDump(F1Metric):
    def __init__(self, pos_label: int = 1, out_file_path: Optional[str] = None, ann_file_path: Optional[str] = None,
                 collect_device: str = 'cpu', prefix: Optional[str] = None) -> None:
        super().__init__(pos_label=pos_label, collect_device=collect_device, prefix=prefix)
        self.out_file_path = out_file_path
        self.ann_file_path = ann_file_path

    def compute_metrics(self, results: List) -> Dict:
        metrics = super().compute_metrics(results)
        if self.out_file_path is None:
            return metrics

        annotations = []
        if self.ann_file_path and Path(self.ann_file_path).exists():
            with open(self.ann_file_path, 'rb') as f:
                annotations = pickle.load(f)

        payload = []
        for idx, result in enumerate(results):
            ann = annotations[idx] if idx < len(annotations) else {}
            score = result.get('pred_score')
            probs = score.numpy().tolist() if isinstance(score, torch.Tensor) else None
            payload.append({
                'sample_id': ann.get('sample_uid', ann.get('sample_id', f'sample_{idx}')),
                'subject': ann.get('subject', 'unknown'),
                'env': ann.get('environment', 'unknown'),
                'trial': ann.get('trial', 'unknown'),
                'lighting': ann.get('lighting_level', 'normal'),
                'true_label': int(result['gt_label']),
                'pred_label': int(result['pred_label']),
                'probs': probs,
                'label_definition': {'0': 'non-falling', '1': 'falling'},
            })

        out_path = Path(self.out_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        return metrics
