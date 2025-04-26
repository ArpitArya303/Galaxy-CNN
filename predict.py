# predict.py

import torch
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y, mean, std

def classify_galaxies(model, list_of_image_paths, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    predictions = []
    with torch.no_grad():
        for img_path in list_of_image_paths:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            predictions.append(pred.item())
    
    return predictions
