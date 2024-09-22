import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import Union, Tuple, Callable
import pandas as pd
import cv2
import torch
import os 

from utils.aug_utils import get_augmentation, apply_augmentation

def get_transform(config, is_train=True):
    if is_train:
        # get_augmentation 함수를 사용하여 증강 얻기
        custom_augmentations = get_augmentation(config)
        
        return A.Compose([
            A.Resize(224, 224),
            custom_augmentations,
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        info_file: str,
        augmented_dir: str,
        augmented_info_file: str,
        transform: Callable,
        use_augmented: bool = True,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.augmented_dir = augmented_dir
        self.transform = transform
        self.is_inference = is_inference
        self.use_augmented = use_augmented
        # Read original data
        self.info_df = pd.read_csv(info_file)
        self.image_paths = self.info_df['image_path'].tolist()
        if self.use_augmented:
            # Read augmented data
            self.augmented_df = pd.read_csv(augmented_info_file)
            self.augmented_image_paths = self.augmented_df['image_path'].tolist()
            self.all_image_paths = self.image_paths + self.augmented_image_paths
        else:
            self.all_image_paths = self.image_paths
        if not self.is_inference:
            if self.use_augmented:
                self.targets = self.info_df['target'].tolist() + self.augmented_df['target'].tolist()
            else:
                self.targets = self.info_df['target'].tolist()
    def __len__(self) -> int:
        return len(self.all_image_paths)
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        if self.use_augmented and index >= len(self.image_paths):
            img_path = os.path.join(self.augmented_dir, self.all_image_paths[index])
        else:
            img_path = os.path.join(self.root_dir, self.all_image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target