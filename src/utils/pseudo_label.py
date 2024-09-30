import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset


def generate_pseudo_labels(model, test_loader, device, threshold):
    model.eval()
    pseudo_labels = []
    confidences = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)
            pseudo_labels.extend(predictions.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
    
    return pseudo_labels, confidences


def update_dataset_with_pseudo_labels(dataset, pseudo_labels: List[int]):
    dataset.update_pseudo_labels(pseudo_labels)

def create_pseudo_label_dataset(train_dataset, test_dataset) -> ConcatDataset:
    return ConcatDataset([train_dataset, test_dataset])

def pseudo_label_training_step(model, train_loader, test_loader, device, config):
    model.eval()
    test_dataset = test_loader.dataset
    threshold = config['training']['confidence_threshold']
    num_classes = config['model']['num_classes']
    
    if test_dataset.is_inference:
        test_dataset.targets = [-1] * len(test_dataset)
        test_dataset.is_inference = False

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if isinstance(batch, list):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)
            
            for j, (conf, pred) in enumerate(zip(confidence, predictions)):
                idx = i * test_loader.batch_size + j
                if idx < len(test_dataset):
                    if conf.item() >= threshold:
                        test_dataset.targets[idx] = pred.item()
                    else:
                        test_dataset.targets[idx] = -1  # 임계값 미만의 예측은 -1로 설정

    # 미분류된 라벨(-1)을 가진 샘플 제거
    valid_indices = [i for i, target in enumerate(test_dataset.targets) if target != -1]
    valid_test_dataset = Subset(test_dataset, valid_indices)

    combined_dataset = ConcatDataset([train_loader.dataset, valid_test_dataset])
    
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )

    confident_count = len(valid_indices)
    print(f"Number of confident pseudo-labeled samples: {confident_count}")
    print(f"Total samples in combined dataset: {len(combined_dataset)}")

    return combined_loader, confident_count