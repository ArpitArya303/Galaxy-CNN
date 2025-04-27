# Galaxy Classification Project
This project involves building a Convolutional Neural Network (CNN) for the classification of galaxy images into five distinct categories. The model is trained on the Galaxy Zoo dataset using PyTorch and implements various best practices such as data augmentation, learning rate scheduling, and early stopping.

This project implements a Convolutional Neural Network (CNN) to classify galaxy images into following five categories:
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

**Network Layers:**

- Conv1 (3 → 32): The first convolutional layer with 3 input channels (RGB images) and 32 output channels. It uses a 3x3 kernel with padding to maintain the spatial dimensions.

- Max Pooling: After each convolutional layer, max pooling is applied with a 2x2 kernel to reduce the spatial dimensions of the feature map by half.

- Conv2 (32 → 64): The second convolutional layer has 32 input channels and 64 output channels, followed by ReLU activation and max pooling.

- Conv3 (64 → 128): The third convolutional layer has 64 input channels and 128 output channels, followed by ReLU activation and max pooling.

- Adaptive Average Pooling: After the convolutional layers, an adaptive average pooling layer is applied, resizing the feature map to a fixed size of 7x7. This ensures that the model can handle variable input image sizes while outputting a consistent feature map size.

**Fully Connected Layers:**

- FC1: A fully connected layer that takes the output of the adaptive pooling and flattens it into a 1D vector, followed by 512 units.

- FC2: The second fully connected layer with 256 units.

- FC3: The final fully connected layer outputs 5 units, corresponding to the 5 galaxy classes.

Activation Functions:

- ReLU: ReLU activation is applied after each convolutional and fully connected layer, which helps introduce non-linearity and allows the network to learn more complex patterns.

Output:

- The output layer (FC3) produces a vector of size 5, representing the 5 different galaxy classes

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

Note: The train and validation was performed on this dataset. The test set was not touched and if you want can check the model performance on it.

**Path Instructions:**
1. Download and extract the dataset.
2. The dataset is organised as:

```

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

Given the relatively small size of the dataset, a moderately deep custom CNN was chosen to balance performance with training efficiency. Pre-trained models were not used to focus on building a model from scratch to better understand the process.

## Interface Summary

All important functions and classes are exposed via `interface.py` for easy access during evaluation.

| Purpose | Interface Name | Source File |
|:--------|:---------------|:------------|
| Model Class | `TheModel` | `model.py` |
| Training Function | `the_trainer` | `train.py` |
| Prediction Function | `the_predictor` | `predict.py` |
| Custom Dataset Class | `TheDataset` | `dataset.py` |
| Data Loader | `the_dataloader` | `dataset.py` |

## Additional Notes

- **Hardware:** Training was accelerated using GPU (CUDA), but it can also run on CPU with longer training time.
- **Early stopping** and **learning rate scheduler** were employed to prevent overfitting.
- **Class imbalance** was handled by computing class weights based on sample counts.
- **Data Augmentation:** Applied horizontal flips, vertical flips, and random rotations to improve generalization.
- **Accuracy:** The model achieved an accuracy of approximately 94% on the validation set.

## Evaluation Notes

- The test set was left untouched for evaluation purposes.
