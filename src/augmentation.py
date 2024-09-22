import pandas as pd
import cv2
import os
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from utils.aug_utils import get_augmentation, apply_augmentation, create_augmented_image_name 

def process_image(row, config, augmentations):
    train_dir = config['data']['train_dir']
    augmented_dir = config['data']['augmented_dir']

    relative_path = row['image_path']
    full_image_path = os.path.join(train_dir, relative_path)
    target = row['target']
    
    class_name = os.path.dirname(relative_path)
    augmented_class_dir = os.path.join(augmented_dir, class_name)
    os.makedirs(augmented_class_dir, exist_ok=True)
    
    image = cv2.imread(full_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    new_rows = []
    for aug_type, aug in enumerate(augmentations):
        augmented_image = apply_augmentation(image, aug)
        
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
    
    augmentations = get_augmentation(config['offline_augmentation'])
    
    df = pd.read_csv(train_info_file)
    
    process_func = partial(process_image, config=config, augmentations=augmentations)
    
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