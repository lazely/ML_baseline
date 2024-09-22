import os
import re
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from typing import Union, Tuple, Callable
from utils.datasets import CustomDataset, get_transform

def get_next_version(directory):
    """
    현재 디렉토리에서 가장 높은 버전 번호를 찾고, 그 다음 번호를 반환합니다.
    """
    version_pattern = re.compile(r'train_info(\d+)\.csv')
    highest_version = 0
    
    # 디렉토리의 모든 파일을 검색하여 버전 번호를 추출
    for filename in os.listdir(directory):
        match = version_pattern.match(filename)
        if match:
            version = int(match.group(1))
            if version > highest_version:
                highest_version = version
    
    # 다음 버전 번호를 반환
    return highest_version + 1

        
def get_test_loaders(config):
    dataset = CustomDataset(
        root_dir=config['data']['test_dir'],
        info_file=config['data']['test_info_file'],
        augmented_dir=config['data']['augmented_dir'],
        augmented_info_file=config['data']['augmented_info_file'],
        transform=get_transform(config,is_train=False),
        is_inference=True,
        use_augmented=False
    )
    test_data_loaders = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )
    return test_data_loaders

def get_data_loaders(config, batch_size=None):
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # 전체 데이터셋 로드 (오프라인 증강 이미지는 사용하지 않음)
    full_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=config['data']['train_info_file'],
        augmented_dir=None,  # 오프라인 증강 이미지 디렉토리 제외
        augmented_info_file=None,  # 오프라인 증강 정보 파일 제외
        transform=get_transform(config, is_train=False),  # 기본 Transform 적용 (증강 X)
        use_augmented=False  # 오프라인 증강 이미지를 사용하지 않음
    )
    
    # 8:2로 train/val 나누기
    train_size = int((1 - config['training']['validation_ratio']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # train/val 나눈 데이터 저장
    train_df = full_dataset.info_df.iloc[train_indices.indices]
    val_df = full_dataset.info_df.iloc[val_indices.indices]

    # 저장 경로 생성
    split_dir = config['data']['split_dir']
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    
    # 다음 버전 번호를 결정
    version = get_next_version(split_dir)
        
    train_save_path = os.path.join(split_dir, f'train_info{version}.csv')
    val_save_path = os.path.join(split_dir, f'val_info{version}.csv')
    
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    
    train_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=os.path.join(config['data']['split_dir'], f'train_info{version}.csv'),  # 버전 관리된 train 데이터
        augmented_dir=config['data']['augmented_dir'],  # 오프라인 증강 이미지 디렉토리 사용
        augmented_info_file=config['data']['augmented_info_file'],  # 오프라인 증강 정보 파일 사용
        transform=get_transform(config, is_train=True),  # 증강 적용
        use_augmented=True  # 오프라인 증강 이미지 포함
    )
    
    val_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=os.path.join(config['data']['split_dir'], f'val_info{version}.csv'),  # 버전 관리된 val 데이터
        augmented_dir=None,  # 오프라인 증강 이미지 미사용
        augmented_info_file=None,  # 오프라인 증강 정보 파일 미사용
        transform=get_transform(config, is_train=False),  # 증강 미적용
        use_augmented=False  # 오프라인 증강 미사용
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
