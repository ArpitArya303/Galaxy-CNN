import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import learning_rate, epochs, early_stop_patience, train_transform, val_test_transform , train_path, val_path, device
from dataset import get_loader , get_dataset
import numpy as np

def train_model(model):

    train_dataset = get_dataset(train_path, transform=train_transform)
    
    # Loss function with class weights
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()  
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_valid_loss = float('inf')
    early_stop_counter = 0

    train_loader = get_loader(train_path, transform=train_transform, shuffle=True)
    valid_loader = get_loader(val_path, transform=val_test_transform, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0

        

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100 * valid_correct / valid_total

        scheduler.step(avg_valid_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%")
    

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'checkpoints/final_weights.pth')
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break
