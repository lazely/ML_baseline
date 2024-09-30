import torch
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm
from src.models.model_utils import get_model, get_ensemble_model
from src.utils.data_loaders import get_test_loaders

def load_model(config, model_path):
    model_name = os.path.basename(model_path).split('_best_model.pth')[0]
    model = get_ensemble_model(config, model_name).to(torch.device(config['device']))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def get_predictions(model, test_loader, device):
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu())
    return torch.cat(predictions)

def ensemble_predictions(predictions_list, ensemble_type):
    # if len(predictions_list) <= 0:
    #     print("Error")
    if ensemble_type == 'soft':
        ensemble_preds = torch.stack(predictions_list).mean(dim=0)
        return ensemble_preds.argmax(dim=1)
    elif ensemble_type == 'hard':
        hard_votes = torch.stack([pred.argmax(dim=1) for pred in predictions_list])
        ensemble_preds, _ = torch.mode(hard_votes, dim=0)
        return ensemble_preds
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")

def run(config):
    device = torch.device(config['device'])
    test_loader = get_test_loaders(config)

    save_dir = config['paths']['save_dir']
    model_files = [f for f in os.listdir(save_dir) if f.endswith('_best_model.pth')]
    # print(model_files)

    predictions_list = []
    for model_file in model_files:
        model_path = os.path.join(save_dir, model_file)
        model = load_model(config, model_path)
        predictions = get_predictions(model, test_loader, device)
        predictions_list.append(predictions)

    ensemble_preds = ensemble_predictions(predictions_list, config['ensemble']['type'])

    test_info = pd.read_csv(config['data']['test_info_file'])
    test_info['target'] = ensemble_preds.numpy()
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    
    output_path = os.path.join(config['paths']['output_dir'], "ensemble_output.csv")
    test_info.to_csv(output_path, index=False)
    # print(f"Ensemble predictions saved to {output_path}")