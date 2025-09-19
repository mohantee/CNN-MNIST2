# CNN-MNIST2

This project implements a Convolutional Neural Network (CNN) for the MNIST digit classification task, with the goal of achieving >99% accuracy while maintaining a minimal number of parameters. The project includes three iterations of model improvement.

## Dataset

The MNIST dataset is used for training and testing:
- 60,000 training images
- 10,000 test images
- Image size: 28x28 pixels (grayscale)
- 10 classes (digits 0-9)

### Data Preprocessing
- Random center crop (22x22) with probability 0.1
- Resize to 28x28
- Random rotation (-15° to +15°)
- Normalization with mean=0.1307 and std=0.3081

## Model Architecture Overview

All iterations share a similar base architecture with two convolutional layers followed by fully connected layers. The main differences are in the use of batch normalization and the number of neurons in the fully connected layers.

## Iterations Comparison

| Feature                | Iteration 1         | Iteration 2         | Iteration 3         |
|------------------------|---------------------|---------------------|---------------------|
| Total Parameters       | ~24,245             | ~24,471             | ~14,538             |
| Test Accuracy          | 98.95%              | 99.52%              | 99.37%   |
| Batch Size             | 512                 | 1024                | 1024                |
| Batch Normalization    | No                  | Yes                 | Yes                 |
| Dropout                | No                  | No                  | No                  |
| MaxPooling             | After each Conv     | After each Conv     | After each Conv     |
| FC Layer Sizes         | 288→32→10           | 288→65→10           | 288→32→10           |
| Epochs                 | 20                  | 20                  | 20                  |
| Model Improvements     | Base Model          | +BatchNorm, +FC     | Fewer params, +BN   |

**Notes:**
- MaxPooling is always after each convolutional layer.
- No Dropout is used; consider adding if overfitting is observed.
- All convolutions are 3x3; no 1x1 convolutions or transition layers.
- Batch normalization is used in Iterations 2 and 3.
- Learning rate is reduced after 15 epochs using StepLR.
- Test accuracy should be filled in after running the notebooks.

### Training Configuration
- Optimizer: Adam (lr=0.01)
- Weight Decay: 1e-4
- Learning Rate Scheduler: StepLR (step_size=15, gamma=0.1)
- Loss Function: CrossEntropyLoss
- Number of Epochs: 20

## Model Architecture Details

### Iteration 1 (Base Model)
```python
Conv2d(1, 16, kernel_size=3)
MaxPool2d(2)
Conv2d(16, 32, kernel_size=3)
MaxPool2d(3)
Linear(288, 32)
Linear(32, 10)
```

### Iteration 2 (Added Batch Normalization)
```python
Conv2d(1, 16, kernel_size=3) -> BatchNorm2d
MaxPool2d(2)
Conv2d(16, 32, kernel_size=3) -> BatchNorm2d
MaxPool2d(3)
Linear(288, 65) -> BatchNorm1d
Linear(65, 10)
```

### Iteration 3 (Optimized Architecture)
```python
Conv2d(1, 16, kernel_size=3) -> BatchNorm2d
MaxPool2d(2)
Conv2d(16, 32, kernel_size=3) -> BatchNorm2d
MaxPool2d(3)
Linear(288, 32) -> BatchNorm1d
Linear(32, 10)
```

#### Architecture Diagram
```
Input (1x28x28)
      │
Conv2d(1,16,3x3) → BatchNorm2d
      │
   ReLU
      │
MaxPool2d(2x2)
      │
Conv2d(16,32,3x3) → BatchNorm2d
      │
   ReLU
      │
MaxPool2d(3x3)
      │
Flatten
      │
Linear(288,32) → BatchNorm1d
      │
   ReLU
      │
Linear(32,10)
      │
LogSoftmax
      │
Output (10 classes)
```

## Key Improvements
1. Iteration 1 → 2:
   - Added batch normalization layers
   - Increased neurons in first FC layer (32 → 65)
   - Increased batch size (512 → 1024)

2. Iteration 2 → 3:
   - Reduced model parameters by ~40%
   - Maintained batch normalization
   - Optimized FC layer architecture

## Running the Models

Each iteration is contained in its own Jupyter notebook:
- `mnist-iteration1.ipynb`: Base model
- `mnist-iteration2.ipynb`: Enhanced model with batch normalization
- `mnist-iteration3.ipynb`: Optimized model architecture

Requirements:
- PyTorch
- torchvision
- matplotlib
- tqdm
- torchsummary

To run any iteration:
1. Open the respective notebook
2. Execute all cells in sequence
3. Monitor training progress with the built-in progress bars
4. View final accuracy and loss metrics

## Iteration 3: Last 2 Epochs Training and Test Logs

Epoch 19
Train: Loss=0.0287 Batch_id=58 Accuracy=99.33: 100%|██████████| 59/59 [00:16<00:00,  3.60it/s]
Test set: Average loss: 0.0000, Accuracy: 59580/60000 (99.30%)

Epoch 20
Train: Loss=0.0316 Batch_id=58 Accuracy=99.33: 100%|██████████| 59/59 [00:17<00:00,  3.42it/s]
Test set: Average loss: 0.0000, Accuracy: 59623/60000 (99.37%)
