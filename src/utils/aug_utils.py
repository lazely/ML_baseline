import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from albumentations import (
    Compose, RandomCrop, CenterCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightnessContrast, HueSaturationValue, 
    GaussNoise, Blur, OpticalDistortion, GridDistortion, 
    ElasticTransform, CoarseDropout, Resize, OneOf
)
def get_augmentation(config):
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

    return Compose(aug_ops)



def apply_augmentation(image, augmentation):
    return augmentation(image=image)['image']

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_augmented_image_name(original_name, aug_type):
    name, ext = os.path.splitext(original_name)
    return f"{name}_aug_{aug_type}{ext}"


