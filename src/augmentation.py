import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from src.utils.aug_utils import get_augmentation, apply_augmentation, apply_augmentation_with_mixup_cutmix, create_augmented_image_name 
from src.utils.aug_utils import MixupTransform, CutMixTransform, mixup, cutmix

def process_image(row, config, augmentations, dataframe):
    train_dir = config['data']['train_dir']
    augmented_dir = config['data']['augmented_dir']

    relative_path = row['image_path']
    full_image_path = os.path.join(train_dir, relative_path)
    target = row['target']
    
    class_name = os.path.dirname(relative_path)
    augmented_class_dir = os.path.join(augmented_dir, class_name)
    os.makedirs(augmented_class_dir, exist_ok=True)
    
    image1 = cv2.imread(full_image_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Randomly select another image for augmentation
    image2_row = dataframe.sample(n=1).iloc[0]
    image2_relative_path = image2_row['image_path']
    image2_full_path = os.path.join(train_dir, image2_relative_path)
    image2 = cv2.imread(image2_full_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    new_rows = []

    for aug_type, aug in enumerate(augmentations):

        if isinstance(aug, MixupTransform):
            augmented_image = mixup(image1, image2)
            
            # mixed_target = lam * target + (1 - lam) * image2_row['target']
            # new_target = mixed_target.round().astype(int)
            new_target = target

        elif isinstance(aug, CutMixTransform):
            augmented_image = cutmix(image1, image2)
            new_target = target

        # label 섞는 것이 의도
        else:
            augmented_image = apply_augmentation(image1, aug)
            new_target = target
        
        original_filename = os.path.basename(full_image_path)
        new_file_name = create_augmented_image_name(original_filename, aug_type)
        new_relative_path = os.path.join(class_name, new_file_name)
        new_full_path = os.path.join(augmented_dir, new_relative_path)
        
        cv2.imwrite(new_full_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        
        new_rows.append({
            'image_path': new_relative_path,
            'target': target
        })
    
    return new_rows

def run(config):
    train_info_file = config['data']['train_info_file']
    augmented_info_file = config['data']['augmented_info_file']
    
    df = pd.read_csv(train_info_file)
    
    augmentations = get_augmentation(config['offline_augmentation'], df)
    
    process_func = partial(process_image, config=config, augmentations=augmentations, dataframe=df)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_func, df.to_dict('records')), 
            total=len(df), 
            desc="Augmenting images"
        ))
    
    new_rows = [item for sublist in results for item in sublist]
    augmented_df = pd.DataFrame(new_rows)
    augmented_df.to_csv(augmented_info_file, index=False)
    
    print(f"Augmentation complete. Number of augmented images: {len(augmented_df)}")