import torch 
from torchvision import datasets
from torch.utils.data import DataLoader
from config import batch_size, num_workers

def get_dataset(path, transform=None):
    # Load the dataset from the specified path with required transformations.
    
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return dataset

def get_loader(path, transform=None, shuffle=False, batch_size=batch_size, num_workers=num_workers):
    # Load the dataloader from the specified path with required transformations.

    dataset = get_dataset(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

