#!/usr/bin/env python3
"""Fusion training for the online retraining workspace."""

import argparse
import glob
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
if '/LCFall' not in sys.path:
    sys.path.insert(0, '/LCFall')
if '/LCFall/src' not in sys.path:
    sys.path.insert(0, '/LCFall/src')
if '/LCFall/external_libs/mmaction2' not in sys.path:
    sys.path.insert(0, '/LCFall/external_libs/mmaction2')

from fusion.model import create_fusion_model
from online_train.fusion_mlp.dataset_online_retrain import OnlineRetrainFusionPoseDataset, collate_fn
from mmaction.registry import DATASETS
from mmengine.config import Config
from mmaction.utils import register_all_modules

register_all_modules()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_manifest_samples(path: str) -> list[dict]:
    with open(path, 'r') as f:
        payload = json.load(f)
    return payload['samples'] if isinstance(payload, dict) else payload


def compute_class_weight(samples: list[dict]) -> list[float]:
    non_falling = sum(1 for sample in samples if int(sample['label']) == 0)
    falling = sum(1 for sample in samples if int(sample['label']) == 1)
    if falling == 0:
        return [1.0, 1.0]
    return [1.0, max(1.0, non_falling / falling)]


def resolve_checkpoint_path(pattern: str) -> str:
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]
    raise FileNotFoundError(f'Not found: {pattern}')


class UnifiedFusionTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module,
                 optimizer: optim.Optimizer, device: torch.device, save_dir: Path, patience: int = 15,
                 min_delta: float = 0.001, pos_label: int = 1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        self.min_delta = min_delta
        self.pos_label = pos_label
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'lr': []
        }
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1} Train')
        for pose_data_list, pointcloud_data, labels, _ in pbar:
            pointcloud_data = pointcloud_data.to(self.device)
            labels = labels.to(self.device)
            skeleton_heatmaps = []
            for pose_data in pose_data_list:
                heatmap = pose_data['inputs']
                if heatmap.ndim == 5:
                    heatmap = heatmap[0]
                skeleton_heatmaps.append(heatmap)
            skeleton_heatmaps = torch.stack(skeleton_heatmaps, dim=0).to(self.device)
            outputs = self.model(skeleton_heatmaps, pointcloud_data)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return running_loss / len(all_labels), accuracy_score(all_labels, all_preds) * 100, f1_score(all_labels, all_preds, pos_label=self.pos_label, zero_division=0) * 100

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pose_data_list, pointcloud_data, labels, _ in tqdm(self.val_loader, desc='Validation'):
                pointcloud_data = pointcloud_data.to(self.device)
                labels = labels.to(self.device)
                skeleton_heatmaps = []
                for pose_data in pose_data_list:
                    heatmap = pose_data['inputs']
                    if heatmap.ndim == 5:
                        heatmap = heatmap[0]
                    skeleton_heatmaps.append(heatmap)
                skeleton_heatmaps = torch.stack(skeleton_heatmaps, dim=0).to(self.device)
                outputs = self.model(skeleton_heatmaps, pointcloud_data)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return {
            'loss': running_loss / len(all_labels),
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'f1_score': f1_score(all_labels, all_preds, pos_label=self.pos_label, zero_division=0) * 100,
            'precision': precision_score(all_labels, all_preds, pos_label=self.pos_label, zero_division=0) * 100,
            'recall': recall_score(all_labels, all_preds, pos_label=self.pos_label, zero_division=0) * 100,
        }

    def train(self, num_epochs: int) -> Dict:
        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            if val_metrics['f1_score'] > self.best_val_f1 + self.min_delta:
                self.best_val_f1 = val_metrics['f1_score']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, 'best_model.pth', val_metrics)
                improved = 'yes'
            else:
                self.patience_counter += 1
                improved = 'no'
            print(
                f"[epoch {epoch + 1}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} train_f1={train_f1:.2f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2f} "
                f"val_f1={val_metrics['f1_score']:.2f} val_precision={val_metrics['precision']:.2f} "
                f"val_recall={val_metrics['recall']:.2f} best_val_f1={self.best_val_f1:.2f} "
                f"improved={improved} patience={self.patience_counter}/{self.patience}",
                flush=True,
            )
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break
        self.save_history()
        self.plot_learning_curves()
        return {'best_f1': self.best_val_f1, 'best_epoch': self.best_epoch, 'elapsed_hours': (time.time() - start_time) / 3600}

    def save_checkpoint(self, epoch: int, filename: str, val_metrics: Dict):
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'val_f1': val_metrics['f1_score'], 'best_val_f1': self.best_val_f1}, self.save_dir / filename)

    def save_history(self):
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump({**self.history, 'best_epoch': self.best_epoch, 'best_val_f1': self.best_val_f1}, f, indent=2)

    def plot_learning_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(self.history['train_f1'], label='Train')
        axes[1].plot(self.history['val_f1'], label='Val')
        axes[1].axvline(self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('F1 Score')
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'learning_curves.png', dpi=150)
        plt.close()


def create_dataloaders(camera_config_path: str, train_manifest_path: str, val_manifest_path: str, batch_size: int = 16, num_workers: int = 0):
    cfg = Config.fromfile(camera_config_path)
    pose_train = DATASETS.build(cfg.train_dataloader.dataset)
    pose_val = DATASETS.build(cfg.val_dataloader.dataset)
    pointcloud_data_dir = Path('/LCFall/src/online_train/generated/lidar_roi_frames')
    train_dataset = OnlineRetrainFusionPoseDataset(pose_train, Path(train_manifest_path), pointcloud_data_dir, 256)
    val_dataset = OnlineRetrainFusionPoseDataset(pose_val, Path(val_manifest_path), pointcloud_data_dir, 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description='online_train Fusion MLP training')
    parser.add_argument('--camera_config', type=str, default='/LCFall/src/online_train/camera/configs/slowonly_r50_unified.py')
    parser.add_argument('--camera_ckpt', type=str, default='/LCFall/src/online_train/camera/checkpoints/train_subject_split/best_f1_f1_score_epoch_*.pth')
    parser.add_argument('--lidar_ckpt', type=str, default='/LCFall/src/online_train/lidar/checkpoints/train_subject_split/best_model.pth')
    parser.add_argument('--train_manifest_path', type=str, default='/LCFall/src/online_train/manifests/fusion_train.json')
    parser.add_argument('--val_manifest_path', type=str, default='/LCFall/src/online_train/manifests/fusion_val.json')
    parser.add_argument('--save_dir', type=str, default='/LCFall/src/online_train/fusion_mlp/checkpoints/train_subject_split')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    camera_ckpt = Path(resolve_checkpoint_path(args.camera_ckpt))
    lidar_ckpt = Path(args.lidar_ckpt)
    if not camera_ckpt.exists():
        raise FileNotFoundError(f'Camera checkpoint not found: {camera_ckpt}')
    if not lidar_ckpt.exists():
        raise FileNotFoundError(f'LiDAR checkpoint not found: {lidar_ckpt}')
    class_weight = compute_class_weight(load_manifest_samples(args.train_manifest_path))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = create_dataloaders(args.camera_config, args.train_manifest_path, args.val_manifest_path, args.batch_size, args.num_workers)
    print({
        'device': str(device),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'class_weight': class_weight,
        'lr': args.lr,
        'dropout': args.dropout,
    }, flush=True)
    model = create_fusion_model(
        args.camera_config,
        str(camera_ckpt),
        str(lidar_ckpt),
        num_classes=2,
        freeze_backbones=True,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32).to(device))
    optimizer = optim.AdamW(model.get_trainable_parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    trainer = UnifiedFusionTrainer(model, train_loader, val_loader, criterion, optimizer, device, save_dir, pos_label=1)
    result = trainer.train(args.num_epochs)
    result['route'] = 'train'
    with open(save_dir / 'config.json', 'w') as f:
        json.dump({
            'route': 'train',
            'camera_config': args.camera_config,
            'camera_ckpt': str(camera_ckpt),
            'lidar_ckpt': str(lidar_ckpt),
            'train_manifest_path': args.train_manifest_path,
            'val_manifest_path': args.val_manifest_path,
            'class_weight': class_weight,
            'lr': args.lr,
            'dropout': args.dropout,
            'label_definition': {'0': 'non-falling', '1': 'falling'}
        }, f, indent=2)
    with open(save_dir / 'training_summary.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
