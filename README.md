# MNIST Neural Network (NumPy Only)

## Project Overview
This project implements a neural network from scratch to classify MNIST handwritten digits with a high accuracy of 98.02%. It is entirely built using only `NumPy` without the use of any machine learning libraries like TensorFlow or PyTorch. The network consists of input, hidden, and output layers with backpropagation to adjust weights over 100 epochs.

## Techniques Used
- **Data Preprocessing**: Flattened images and normalized pixel values to ensure efficient training.
- **Neural Network Layers**: Implemented a 3-layer neural network with one hidden layer of 128 units.
- **Activation Function**: Used the sigmoid activation function for both hidden and output layers.
- **Backpropagation**: Calculated error gradients and updated weights to minimize the loss function.
- **Gradient Descent**: Used a fixed learning rate of 0.1 to optimize the model over time.

## Results
- **Training Accuracy**: 98.02%
- **Loss**: Converges to 0.0007 after 40 epochs





