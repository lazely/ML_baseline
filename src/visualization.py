import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchcam.methods import GradCAM
import os
from src.models.model_utils import get_model
from src.utils.data_loaders import get_data_loaders

def visualize_gradcam(
        model: torch.nn.Module,
        device: torch.device,
        target_layer: str,
        image_index: int,
        config
    ):
    train_loader, val_loader = get_data_loaders(config)
    # 초기화
    cam_extractor = GradCAM(model, target_layer)

    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        inputs = inputs.to(device) 

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size(0)):

            try:
                cam = cam_extractor(preds[j].item(), outputs[j].unsqueeze(0), retain_graph=True)[0]
                # cam 계산 시 그래프 유지 해야 오류X
                
            except RuntimeError as e:
                print(f"Error generating Grad-CAM for image {batch_idx * len(inputs) + j}: {e}")
                continue

            # CAM을 1채널로 변환
            cam = cam.mean(dim=0).cpu().numpy()

            # CAM을 원본 이미지 크기로
            cam = cv2.resize(cam, (inputs[j].shape[2], inputs[j].shape[1]))

            # CAM을 정규화
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # +1e-8을 추가하여 0으로 나누는 것을 방지

            # CAM을 0-255 범위로 변환
            cam = np.uint8(255 * cam)

            # 컬러맵을 적용하여 RGB 이미지로 변환
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

            # 입력 이미지가 1채널 또는 3채널인지 확인하고 처리
            input_image = inputs[j].cpu().numpy().transpose((1, 2, 0))

            if input_image.shape[2] == 1:  # 1채널 이미지인 경우
                input_image = np.squeeze(input_image, axis=2)  # (H, W, 1) -> (H, W)
                input_image = np.stack([input_image] * 3, axis=-1)  # (H, W) -> (H, W, 3)로 변환하여 RGB처럼 만듭니다.

            else:  # 3채널 이미지인 경우
                input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
                input_image = (input_image * 255).astype(np.uint8)  # 정규화된 이미지를 8비트 이미지로 변환

            # 시각화용 폴더 생성
            visualization_dir = config['paths']['visualization_dir']
            os.makedirs(visualization_dir, exist_ok=True)

            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title(f"Original Image {batch_idx * len(inputs) + j}")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(cam)
            plt.title(f"Grad-CAM Image {batch_idx * len(inputs) + j}")
            plt.axis('off')

            overlay = cv2.addWeighted(input_image, 0.5, cam, 0.5, 0)
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title(f"Overlay Image {batch_idx * len(inputs) + j}")
            plt.axis('off')

            plt.savefig(f"{visualization_dir}/visualization_image_{batch_idx * len(inputs) + j}.png")  # 시각화 저장
            plt.close()

def run(config):
    device = torch.device(config['device'])

    model = get_model(config).to(device)
    model.load_state_dict(torch.load(f"{config['paths']['save_dir']}/final_model1.pth"))
    model.eval()

    test_loader, val_loader = get_data_loaders(config)
    
    # 전체 구조를 확인하여 레이어 이름 찾기
    # for name, module in model.named_modules():
    # print(name)
    
    target_class_index = 0 
    visualize_gradcam(model, device, 'stages.3.blocks.0', target_class_index, config) # 타겟 레이어 이름 변경가능