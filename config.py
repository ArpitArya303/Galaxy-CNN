import torch
import torchvision.transforms as transforms

# Image 
resize_x = 224
resize_y = 224
input_channels = 3

# Dataset normalization
mean = [0.0461, 0.0405, 0.0299]
std = [0.0831, 0.0696, 0.0586]

# Training settings
batch_size = 64  
learning_rate = 0.001  
num_workers = 4 
num_epochs = 100
early_stop_patience = 10
num_classes = 5

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using GPU: Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")
device = device

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean , std=std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Paths to dataset directories
train_path = 'data'   # change this to your training data path
val_path = 'data'   # change this to your validation data path
test_path = 'data'  # change this to your test data path

# weights path
weights_path = 'checkpoints/final_weights.pth'  # change this to your new weights path if trained again

