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
        df = pd.read_csv(config['data']['train_info_file'])
        custom_augmentations = get_augmentation(config, df)
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
        transform: Callable,
        augmented_dir: str = None,
        augmented_info_file: str = None,
        use_augmented: bool = False,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.augmented_dir = augmented_dir
        self.transform = transform
        self.is_inference = is_inference
        self.use_augmented = use_augmented

        self.info_df = pd.read_csv(info_file)
        self.image_paths = self.info_df['image_path'].tolist()
        self.sketch_name = [path.split(".")[0] for path in self.image_paths]
        
        if self.use_augmented and augmented_info_file:
            self.augmented_df = pd.read_csv(augmented_info_file)
            original_files = {path: True for path in self.image_paths}
            # Augmented 이미지 필터링 (벡터화 연산 사용)
            self.augmented_df['original_path'] = self.augmented_df['image_path'].apply(
                lambda x: os.path.join(os.path.dirname(x), os.path.splitext(os.path.basename(x))[0].split('_aug')[0] + '.JPEG')
            )
            self.augmented_df = self.augmented_df[self.augmented_df['original_path'].isin(original_files)]
                    
            self.augmented_image_paths = self.augmented_df['image_path'].tolist()
            self.all_image_paths = self.image_paths + self.augmented_image_paths
        else:
            self.all_image_paths = self.image_paths

        if not self.is_inference:
            if self.use_augmented and augmented_info_file:
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
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target
