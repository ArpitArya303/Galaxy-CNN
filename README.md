# Galaxy Image Classification Using CNN
The model has been trained on a galaxy image dataset using PyTorch and demonstrates standard deep learning practices such as data augmentation, learning rate scheduling, and early stopping.

This project implements a Convolutional Neural Network (CNN) to classify galaxy images into five categories:
- Cigar-shaped smooth
- Completely round smooth
- Edge-on
- In-between smooth
- Spiral
  
## Project Description

The goal of this project is to develop an image classifier capable of accurately identifying different types of galaxies from images. To achieve this, a custom CNN was designed with multiple convolutional layers, max pooling, and fully connected layers, and was trained using an augmented version of the original dataset.

This project covers:
- Building and training a CNN from scratch (no transfer learning used)
- Applying data augmentation to enhance the dataset
- Using weighted loss functions to handle class imbalance
- Employing learning rate scheduling and early stopping to optimize training

## Model Architecture

The model, **GalaxyCNN**, consists of:
- 3 convolutional layers with ReLU activation and max-pooling
- Adaptive average pooling
- 3 fully connected (dense) layers
- Output layer for 5 galaxy classes

Key features:
- **Input Size:** 224 × 224 pixels
- **Optimizer:** Adam (learning rate 0.001)
- **Loss Function:** Weighted Cross-Entropy Loss
- **Scheduler:** ReduceLROnPlateau (monitors validation loss)
- **Early Stopping:** Stops if validation loss does not improve for 10 consecutive epochs.

## Dataset

The model is trained on a dataset of galaxy images, organized into 5 classes.

You can download the original dataset from here(This has only one folder named Train_images):

**[Galaxy Zoo Classification](https://www.kaggle.com/datasets/anjosut/galaxy-zoo-classification)**

Or if you prefer using my dataset where I have divided into train, validation and test sets:

**[Galaxy-zoo-split](https://www.kaggle.com/datasets/arpitarya03/galaxy-zoo-split)**

**Instructions:**
1. Download and extract the dataset.
2. The dataset is organised as:

```
dataset/
  Train_images_final/
    Cigar-shaped smooth/
    completely round smooth/
    edge-on/
    In-between smooth/
    spiral/
  Valid_images/
    Cigar-shaped smooth/
    completely round smooth/
    edge-on/
    In-between smooth/
    spiral/
  Test_images/
    Cigar-shaped smooth/
    completely round smooth/
    edge-on/
    In-between smooth/
    spiral/
```

3. Update the `train_path` , `val_path` and `test_path` variables in the code with your local dataset paths.

## Model Choice Rationale

- Simplicity and Control: A custom CNN provides full control over the network architecture, helping better understand the effects of each layer.

- Dataset Size: The galaxy dataset is not extremely large, making a moderately deep CNN sufficient without the need for very deep or pre-trained models.

- Feature Learning: Galaxy features like spiral arms and bulges can be effectively captured by convolutional layers, which specialize in learning local patterns.

- Efficiency: The model is lightweight, allowing it to train quickly even with limited computational resources.

- Educational Purpose: Building the model from scratch reinforces core deep learning concepts, providing valuable learning experience compared to using pre-trained networks.
  
## Outputs



## Additional Notes

- **Hardware:** Training was accelerated using GPU (CUDA), but it can also run on CPU with longer training time.
- **Early stopping** and **learning rate scheduler** were employed to prevent overfitting.
- **Class imbalance** was handled by computing class weights based on sample counts.
- **Data Augmentation:** Applied horizontal flips, vertical flips, and random rotations to improve generalization.

