import torch
from config import device, weights_path, val_test_transform
from PIL import Image
from model import GalaxyCNN  
from dataset import get_dataset

# Load class labels
dataset = get_dataset('data')
class_labels = dataset.classes

# Load model weights 
model = GalaxyCNN()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

def inferloader(list_of_img_paths, transform=val_test_transform):
    images = []
    for path in list_of_img_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)
    Batch = torch.stack(images)
    return Batch 

def classify_galaxies(list_of_img_paths):
    galaxy_batch = inferloader(list_of_img_paths)
    galaxy_batch = galaxy_batch.to(device)

    # Predict
    with torch.no_grad():
        logits = model(galaxy_batch)    
        preds = torch.argmax(logits, dim=1)
        labels = [class_labels[p.item()] for p in preds]

    return labels
