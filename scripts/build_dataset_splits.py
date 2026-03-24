#!/usr/bin/env python3
"""Build subject-disjoint train/val/test manifests for online retraining."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

MANIFEST_DIR = Path('/LCFall/src/online_train/manifests')
SPLIT_CONFIG_PATH = MANIFEST_DIR / 'split_config.json'
SOURCE_MANIFESTS = {
    'camera': MANIFEST_DIR / 'all_samples_camera.json',
    'lidar': MANIFEST_DIR / 'all_samples_lidar.json',
    'fusion': MANIFEST_DIR / 'all_samples_fusion.json',
}


def load_json(path: Path) -> Dict:
    with open(path, 'r') as file:
        return json.load(file)


def filter_samples(samples: List[Dict], subjects: List[str]) -> List[Dict]:
    subject_set = set(subjects)
    return [sample for sample in samples if sample['subject'] in subject_set]


def summarize_split(samples: List[Dict], subjects: List[str]) -> Dict:
    label_counts = Counter(sample['label'] for sample in samples)
    falling = label_counts.get(1, 0)
    non_falling = label_counts.get(0, 0)
    falling_weight = round(non_falling / falling, 4) if falling > 0 else 1.0
    return {
        'subjects': subjects,
        'samples': len(samples),
        'label_counts': {
            '0_non_falling': non_falling,
            '1_falling': falling,
        },
        'recommended_class_weight': [1.0, max(1.0, falling_weight)],
    }


def write_manifest(modality: str, split_name: str, source_payload: Dict, samples: List[Dict]) -> Dict:
    payload = dict(source_payload)
    payload['split_name'] = split_name
    payload['samples'] = samples
    output_path = MANIFEST_DIR / f'{modality}_{split_name}.json'
    with open(output_path, 'w') as file:
        json.dump(payload, file, indent=2)
    return summarize_split(samples, payload.get(f'{split_name}_subjects', []))


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    split_config = load_json(SPLIT_CONFIG_PATH)
    split_summary: Dict[str, Dict] = {
        'strategy': split_config['strategy'],
        'description': split_config.get('description', ''),
        'train_subjects': split_config['train_subjects'],
        'val_subjects': split_config['val_subjects'],
        'test_subjects': split_config['test_subjects'],
    }

    for modality, manifest_path in SOURCE_MANIFESTS.items():
        source_payload = load_json(manifest_path)
        source_payload['train_subjects'] = split_config['train_subjects']
        source_payload['val_subjects'] = split_config['val_subjects']
        source_payload['test_subjects'] = split_config['test_subjects']
        samples = source_payload['samples']

        train_samples = filter_samples(samples, split_config['train_subjects'])
        val_samples = filter_samples(samples, split_config['val_subjects'])
        test_samples = filter_samples(samples, split_config['test_subjects'])

        split_summary[modality] = {
            'train': write_manifest(modality, 'train', source_payload, train_samples),
            'val': write_manifest(modality, 'val', source_payload, val_samples),
            'test': write_manifest(modality, 'test', source_payload, test_samples),
        }

    with open(MANIFEST_DIR / 'split_summary.json', 'w') as file:
        json.dump(split_summary, file, indent=2)

    print(json.dumps(split_summary, indent=2))


if __name__ == '__main__':
    main()
