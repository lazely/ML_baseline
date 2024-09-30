import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, Any, Tuple
from utils.datasets import CustomDataset, get_transform

def split_and_save_dataset(config: Dict[str, Any]) -> Tuple[str, str]:
    full_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=config['data']['train_info_file'],
        transform=get_transform(config, is_train=False),
        use_augmented=False
    )
    
    total_size = len(full_dataset)
    train_size = int((1 - config['training']['validation_ratio']) * total_size)
    val_size = total_size - train_size
    
    indices = torch.randperm(total_size).tolist()
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    split_dir = config['data']['split_dir']
    os.makedirs(split_dir, exist_ok=True)
    
    train_save_path = os.path.join(split_dir, 'train_info.csv')
    val_save_path = os.path.join(split_dir, 'val_info.csv')
    
    full_dataset.info_df.iloc[train_indices].to_csv(train_save_path, index=False)
    full_dataset.info_df.iloc[val_indices].to_csv(val_save_path, index=False)
    
    return train_save_path, val_save_path

def get_data_loaders(config: Dict[str, Any], batch_size: int = None) -> Tuple[DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    train_info_file, val_info_file = split_and_save_dataset(config)
    
    train_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=train_info_file,
        augmented_dir=config['data']['augmented_dir'],
        augmented_info_file=config['data']['augmented_info_file'],
        transform=get_transform(config, is_train=True),
        use_augmented=True
    )
    
    val_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=val_info_file,
        transform=get_transform(config, is_train=False),
        use_augmented=False
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

def get_test_loaders(config: Dict[str, Any]) -> DataLoader:
    test_dataset = CustomDataset(
        root_dir=config['data']['test_dir'],
        info_file=config['data']['test_info_file'],
        transform=get_transform(config, is_train=False),
        is_inference=True,
        use_augmented=False
    )
    return DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )