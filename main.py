import yaml
from pathlib import Path
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np

from src import train
from src import test 
from src import optimization
from src import augmentation

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config():
    config = {}
    config_files = ['base.yaml', 'data_model.yaml', 'training.yaml', 'augmentation_optimization.yaml']
    for file in config_files:
        with open(f'configs/{file}', 'r') as f:
            config.update(yaml.safe_load(f))

    if config['training'] == 'cuda' and not torch.cuda.is_available():
        config['training'] = 'cpu'
    
    return config


if __name__ == "__main__":
    config = get_config('base')


    set_random_seed[config['random_seed']]
    mode = config['mode']

    if mode == 'train':
        train.run(config)
    elif mode == 'test':
        test.run(config)
    elif mode == 'augmentation':
        augmentation.run(config)
    elif mode == 'hyperparameter_tune':
        optimization.run(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")


