# predict.py

import torch
from PIL import Image
from model import GalaxyCNN 
from config import test_path, device, val_test_transform
from torchvision import datasets

def predict_galaxy(image_path, model_path):

    # Load dataset to get class labels
    dataset = datasets.ImageFolder(root=test_path)
    class_labels = dataset.classes

    # Load model
    model = GalaxyCNN(num_classes=len(class_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = val_test_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

    print(f"Predicted class for {image_path}: {predicted_class}")
