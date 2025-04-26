# dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import resize_x, resize_y, mean, std, batch_size, num_workers

def get_dataset(path):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return dataset

def get_loader(path):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
