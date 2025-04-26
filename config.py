# config.py

# Image dimensions
resize_x = 224
resize_y = 224
input_channels = 3

# Dataset normalization
mean = [0.0461, 0.0405, 0.0299]
std = [0.0831, 0.0696, 0.0586]

# Training hyperparameters
batch_size = 64
learning_rate = 0.001
num_workers = 4

# Training settings
epochs = 30
early_stop_patience = 10

# Number of classes
num_classes = 5
