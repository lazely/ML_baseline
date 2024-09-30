import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from functools import partial
from albumentations import ImageOnlyTransform
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, HueSaturationValue,
    GaussNoise, Blur, OpticalDistortion, GridDistortion,
    ElasticTransform, CoarseDropout, Rotate, OneOf, Lambda
)
train_dir = "./data/train"

class MixupTransform(ImageOnlyTransform):
    def __init__(self, dataframe, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.dataframe = dataframe

    def apply(self, image, **params):
        # 랜덤으로 다른 이미지를 선택
        image2_row = self.dataframe.sample(n=1).iloc[0]
        image2_path = os.path.join(train_dir, image2_row['image_path'])
        image2 = cv2.imread(image2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        return mixup(image, image2)  # 실제 mixup 함수 호출

class CutMixTransform(ImageOnlyTransform):
    def __init__(self, dataframe, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.dataframe = dataframe

    def apply(self, image, **params):
        # 랜덤으로 다른 이미지를 선택
        image2_row = self.dataframe.sample(n=1).iloc[0]
        image2_path = os.path.join(train_dir, image2_row['image_path'])
        image2 = cv2.imread(image2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        return cutmix(image, image2)  # 실제 cutmix 함수 호출


def get_augmentation(config, dataframe):
    aug_ops = []
    aug_dict = config['augmentation']

    for aug_name, aug_prob in aug_dict.items():
        if aug_name == 'crop':
            aug_ops.append(RandomCrop(height=150, width=150, p=aug_prob))
        elif aug_name == 'flip':
            aug_ops.append(OneOf([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5)
            ], p=aug_prob))
        elif aug_name == 'brightness_contrast':
            aug_ops.append(RandomBrightnessContrast(p=aug_prob))
        elif aug_name == 'hue_saturation':
            aug_ops.append(HueSaturationValue(p=aug_prob))
        elif aug_name == 'noise':
            aug_ops.append(GaussNoise(p=aug_prob))
        elif aug_name == 'blur':
            aug_ops.append(Blur(blur_limit=7, p=aug_prob))
        elif aug_name == 'distortion':
            aug_ops.append(OneOf([
                OpticalDistortion(p=1.0),
                GridDistortion(p=1.0),
                ElasticTransform(p=1.0)
            ], p=aug_prob))
        elif aug_name == 'mask':
            aug_ops.append(CoarseDropout(max_holes=8, max_height=32, max_width=32, p=aug_prob))
        elif aug_name == 'rotation':
            aug_ops.append(Rotate(limit=45, p=aug_prob))

        elif aug_name == 'cutmix':
            aug_ops.append(CutMixTransform(dataframe=dataframe, p=aug_dict['cutmix']))

        elif aug_name == 'mixup':
            aug_ops.append(MixupTransform(dataframe=dataframe, p=aug_dict['mixup']))

    return Compose(aug_ops)


def apply_augmentation(image, augmentation):
    return augmentation(image=image)['image']

def cutmix(image1, image2, alpha=1.0):
    image1 = cv2.resize(image1, (224, 224))
    image2 = cv2.resize(image2, (224, 224))

    h, w = image1.shape[:2]
    lam = np.random.beta(5.0, 2.0)

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    image_mixed = image1.copy()
    image_mixed[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

    return image_mixed

def mixup(image1, image2, alpha=1.0):
    lam = np.random.beta(5.0, 2.0)

    image1 = cv2.resize(image1, (224, 224))
    image2 = cv2.resize(image2, (224, 224))

    image_mixed = lam * image1 + (1 - lam) * image2
    return image_mixed.astype(np.uint8)

def apply_augmentation_with_mixup_cutmix(image1, image2, augmentations, config):
    augmented_images = []

    cutmix_prob = config['augmentation'].get('cutmix', 0)
    mixup_prob = config['augmentation'].get('mixup', 0)

    for aug in augmentations:
        if np.random.random() < cutmix_prob:
            augmented_image = cutmix(image1, image2)
        
        elif np.random.random() < mixup_prob:
            augmented_image =  mixup(image1, image2)
        
        else:
            augmented_image = apply_augmentation(image1, aug)

        augmented_images.append(augmented_image)

    return augmented_images

'''
def apply_augmentation_with_mixup_cutmix(image1, image2, augmentations, config):
    image1_aug = apply_augmentation(image1, augmentations)
    # image2_aug = apply_augmentation(image2, augmentations)

    cutmix_prob = config['augmentation'].get('cutmix', 0)
    mixup_prob = config['augmentation'].get('mixup', 0)

    if np.random.random() < cutmix_prob:
        return cutmix(image1, image2)
    
    if np.random.random() < mixup_prob:
        return mixup(image1, image2)

    return image1_aug
'''

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_augmented_image_name(original_name, aug_type):
    name, ext = os.path.splitext(original_name)
    return f"{name}_aug_{aug_type}{ext}"
