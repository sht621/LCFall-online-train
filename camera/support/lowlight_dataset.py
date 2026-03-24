#!/usr/bin/env python3
"""
低照度データセット用カスタムPoseDatasetおよびTransform

主な機能:
1. FilteredPoseDataset: 有効スケルトンフレームが少ないサンプルをフィルタリング
2. SafeGeneratePoseTarget: 全フレームが0のサンプルでもエラーを発生させない

使用方法:
    configファイルで以下のようにregister:
    
    custom_imports = dict(
        imports=['src.camera.loso_training.lowlight_dataset'],
        allow_failed_imports=False)
    
    dataset=dict(
        type='FilteredPoseDataset',
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        valid_ratio=0.01,  # 少なくとも1%のフレームに有効なスケルトンが必要
        ...)

Author: AI Assistant
Date: 2026-01-13
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable
import mmengine
from mmengine.logging import MMLogger

from mmaction.registry import DATASETS, TRANSFORMS
from mmaction.datasets import PoseDataset
from mmaction.datasets.transforms.pose_transforms import GeneratePoseTarget


@DATASETS.register_module()
class FilteredPoseDataset(PoseDataset):
    """
    有効スケルトンフレーム数に基づいてサンプルをフィルタリングするPoseDataset
    
    Args:
        ann_file: アノテーションファイルパス
        pipeline: データ変換パイプライン
        valid_ratio: 必要な有効フレームの最小割合 (0.0-1.0)
                     例: 0.01 = 1%のフレームに有効なスケルトンが必要
                     Noneまたは0.0の場合はフィルタリングしない
        min_valid_frames: 必要な有効フレームの最小数
                          例: 1 = 少なくとも1フレームに有効なスケルトンが必要
                          Noneまたは0の場合はチェックしない
        filter_mode: 'train'の場合のみフィルタリング、'all'の場合は常にフィルタリング
        **kwargs: PoseDatasetに渡す追加引数
    """
    
    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[Dict, Callable]],
                 valid_ratio: Optional[float] = None,
                 min_valid_frames: Optional[int] = 1,
                 filter_mode: str = 'train',
                 test_mode: bool = False,
                 **kwargs) -> None:
        
        self.valid_ratio_threshold = valid_ratio
        self.min_valid_frames = min_valid_frames
        self.filter_mode = filter_mode
        self._test_mode = test_mode
        self.filtered_count = 0
        
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            **kwargs)
    
    def _count_valid_frames(self, keypoint: np.ndarray) -> int:
        """有効なフレーム数をカウント（0でないキーポイントを持つフレーム）"""
        if keypoint is None:
            return 0
        
        # キーポイント形式を判定
        if keypoint.ndim == 4:
            # (M, T, V, C) 形式
            # 各フレームについて、少なくとも1つのジョイントが非ゼロならValid
            valid_mask = np.any(np.abs(keypoint) > 1e-5, axis=(0, 2, 3))
        elif keypoint.ndim == 3:
            # (T, V, C) 形式
            valid_mask = np.any(np.abs(keypoint) > 1e-5, axis=(1, 2))
        else:
            return 0
        
        return int(np.sum(valid_mask))
    
    def _should_filter(self, data_info: Dict) -> bool:
        """このサンプルをフィルタリングすべきかどうかを判定"""
        # テストモードでfilter_mode='train'の場合はフィルタリングしない
        if self.filter_mode == 'train' and self._test_mode:
            return False
        
        keypoint = data_info.get('keypoint')
        if keypoint is None:
            return True
        
        total_frames = data_info.get('total_frames', 
                                      keypoint.shape[1] if keypoint.ndim == 4 else keypoint.shape[0])
        valid_frames = self._count_valid_frames(keypoint)
        
        # min_valid_framesチェック
        if self.min_valid_frames is not None and self.min_valid_frames > 0:
            if valid_frames < self.min_valid_frames:
                return True
        
        # valid_ratioチェック
        if self.valid_ratio_threshold is not None and self.valid_ratio_threshold > 0:
            if total_frames > 0:
                ratio = valid_frames / total_frames
                if ratio < self.valid_ratio_threshold:
                    return True
        
        return False
    
    def load_data_list(self) -> List[Dict]:
        """アノテーションファイルをロードし、フィルタリングを適用"""
        # 親クラスのload_data_listを呼び出し
        data_list = super().load_data_list()
        
        # フィルタリングが不要な場合はそのまま返す
        should_filter = (
            (self.min_valid_frames is not None and self.min_valid_frames > 0) or
            (self.valid_ratio_threshold is not None and self.valid_ratio_threshold > 0)
        )
        
        if not should_filter:
            return data_list
        
        # テストモードでfilter_mode='train'の場合はフィルタリングしない
        if self.filter_mode == 'train' and self._test_mode:
            return data_list
        
        # フィルタリング実行
        original_count = len(data_list)
        filtered_list = []
        filtered_samples = []
        
        for item in data_list:
            if not self._should_filter(item):
                filtered_list.append(item)
            else:
                filtered_samples.append(item)
        
        self.filtered_count = len(filtered_samples)
        
        # ログ出力
        try:
            logger = MMLogger.get_current_instance()
            logger.info(
                f'FilteredPoseDataset: {original_count} samples loaded, '
                f'{self.filtered_count} filtered out, '
                f'{len(filtered_list)} remaining '
                f'(valid_ratio={self.valid_ratio_threshold}, '
                f'min_valid_frames={self.min_valid_frames}, '
                f'filter_mode={self.filter_mode}, '
                f'test_mode={self._test_mode})'
            )
            
            # フィルタリングされたサンプルの詳細をログ（最初の10件）
            if filtered_samples:
                logger.info(f'Filtered samples (first 10):')
                for i, sample in enumerate(filtered_samples[:10]):
                    frame_dir = sample.get('frame_dir', 'unknown')
                    lighting = sample.get('lighting_level', 'unknown')
                    total = sample.get('total_frames', 0)
                    valid = self._count_valid_frames(sample.get('keypoint'))
                    logger.info(f'  [{i+1}] {frame_dir} ({lighting}): {valid}/{total} valid frames')
                if len(filtered_samples) > 10:
                    logger.info(f'  ... and {len(filtered_samples) - 10} more')
        except Exception:
            # ロガーが使えない場合はprint
            print(f'FilteredPoseDataset: {original_count} -> {len(filtered_list)} samples')
        
        return filtered_list


@TRANSFORMS.register_module()
class SafeGeneratePoseTarget(GeneratePoseTarget):
    """
    全フレームが0のサンプルでも安全に処理できるGeneratePoseTarget
    
    通常のGeneratePoseTargetを継承し、エラーが発生した場合に
    空のヒートマップを返すようにする。
    """
    
    def __init__(self, 
                 fill_value: float = 0.0,
                 handle_all_zeros: bool = True,
                 **kwargs) -> None:
        """
        Args:
            fill_value: エラー時に使用する充填値
            handle_all_zeros: 全てゼロのキーポイントを特別に処理するか
            **kwargs: GeneratePoseTargetに渡す追加引数
        """
        super().__init__(**kwargs)
        self.fill_value = fill_value
        self.handle_all_zeros = handle_all_zeros
    
    def _check_all_zeros(self, results: Dict) -> bool:
        """キーポイントが全てゼロかどうかをチェック"""
        keypoint = results.get('keypoint')
        if keypoint is None:
            return True
        
        return np.all(np.abs(keypoint) < 1e-5)
    
    def transform(self, results: Dict) -> Dict:
        """
        安全なヒートマップ生成
        
        全キーポイントがゼロの場合やエラーが発生した場合は、
        ゼロで埋められたヒートマップを返す。
        """
        try:
            # 全てゼロのキーポイントをチェック
            if self.handle_all_zeros and self._check_all_zeros(results):
                return self._generate_empty_heatmap(results)
            
            # 通常の処理
            return super().transform(results)
            
        except Exception as e:
            # エラーが発生した場合は空のヒートマップを生成
            try:
                logger = MMLogger.get_current_instance()
                frame_dir = results.get('frame_dir', 'unknown')
                logger.warning(
                    f'SafeGeneratePoseTarget: Error processing {frame_dir}: {e}. '
                    f'Returning empty heatmap.'
                )
            except Exception:
                pass
            
            return self._generate_empty_heatmap(results)
    
    def _generate_empty_heatmap(self, results: Dict) -> Dict:
        """空のヒートマップを生成"""
        # 必要な情報を取得
        keypoint = results.get('keypoint')
        img_h, img_w = results.get('img_shape', (64, 64))
        
        # フレーム数を決定
        if keypoint is not None:
            if keypoint.ndim == 4:
                num_frame = keypoint.shape[1]
            elif keypoint.ndim == 3:
                num_frame = keypoint.shape[0]
            else:
                num_frame = 1
        else:
            num_frame = results.get('total_frames', 1)
        
        # チャンネル数を計算
        num_c = 0
        if self.with_kp:
            num_kp = 17  # COCO format
            if keypoint is not None:
                if keypoint.ndim == 4:
                    num_kp = keypoint.shape[2]
                elif keypoint.ndim == 3:
                    num_kp = keypoint.shape[1]
            num_c += num_kp
        if self.with_limb:
            num_c += len(self.skeletons)
        
        # スケーリングを適用
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        
        # 空のヒートマップを生成
        heatmap = np.full([num_frame, num_c, img_h, img_w], 
                          self.fill_value, dtype=np.float32)
        
        # doubleモードの場合は2倍にする
        if self.double:
            heatmap = np.concatenate([heatmap, heatmap])
        
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'
        results[key] = heatmap
        
        return results


@TRANSFORMS.register_module()
class ConvertKeypointFormat(object):
    """
    キーポイント形式を変換するTransform
    
    (T, V, C) -> (M, T, V, C) への変換を行う
    """
    
    def __init__(self, num_person: int = 1):
        """
        Args:
            num_person: 人数（デフォルト: 1）
        """
        self.num_person = num_person
    
    def __call__(self, results: Dict) -> Dict:
        keypoint = results.get('keypoint')
        if keypoint is None:
            return results
        
        # 既に4次元の場合は何もしない
        if keypoint.ndim == 4:
            return results
        
        # (T, V, C) -> (M, T, V, C)
        if keypoint.ndim == 3:
            # T, V, C = keypoint.shape
            keypoint = keypoint[np.newaxis, ...]  # (1, T, V, C)
            
            # 複数人に拡張が必要な場合
            if self.num_person > 1:
                keypoint = np.tile(keypoint, (self.num_person, 1, 1, 1))
                # 追加した人はゼロで埋める
                keypoint[1:] = 0
            
            results['keypoint'] = keypoint
            
            # keypoint_scoreも変換
            if 'keypoint_score' in results:
                kpscore = results['keypoint_score']
                if kpscore.ndim == 2:  # (T, V)
                    kpscore = kpscore[np.newaxis, ...]
                    if self.num_person > 1:
                        kpscore = np.tile(kpscore, (self.num_person, 1, 1))
                        kpscore[1:] = 0
                    results['keypoint_score'] = kpscore
        
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_person={self.num_person})'


@TRANSFORMS.register_module()
class EnsureKeypoint4D(object):
    """
    キーポイントが4次元であることを保証するTransform
    
    3次元 (T, V, C) の場合は (1, T, V, C) に変換
    また、keypoint_scoreを分離している場合は再結合も行う
    """
    
    def __call__(self, results: Dict) -> Dict:
        keypoint = results.get('keypoint')
        if keypoint is None:
            return results
        
        # 3次元 -> 4次元
        if keypoint.ndim == 3:
            keypoint = keypoint[np.newaxis, ...]  # (1, T, V, C)
            results['keypoint'] = keypoint
        
        # keypoint_scoreの処理
        if 'keypoint_score' in results:
            kpscore = results['keypoint_score']
            if kpscore.ndim == 2:  # (T, V) -> (1, T, V)
                results['keypoint_score'] = kpscore[np.newaxis, ...]
        
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

