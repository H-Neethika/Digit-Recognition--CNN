# Handwritten Digit Recognition System

## Overview
This project implements a handwritten digit recognition system using advanced machine learning algorithms. Two approaches have been employed: a Simple Model and a Deep Learning Model utilizing Convolutional Neural Networks (CNNs).

## Data Preprocessing
- The MNIST dataset is loaded and processed.
- Pixel values are normalized within the range [0, 1].
- Labels are transformed into one-hot encoding.
- TensorFlow datasets are constructed for training and testing.

## Simple Model
- A simple neural network model is designed using TensorFlow.
- Components such as forward pass, activation functions, and softmax calculation are implemented.
- The cross-entropy loss function is defined.
- Training is executed utilizing gradient descent optimization.

## Deep Learning Model (CNN)
- A Convolutional Neural Network (CNN) architecture is crafted, incorporating convolutional layers, max-pooling layers, fully connected layers, dropout, and a softmax output layer.
- The loss function is set as cross-entropy.
- Training of the CNN model is conducted employing the Adam optimizer.

## Training and Evaluation
- Both the simple neural network and CNN model undergo training.
- Evaluation is performed on both training and test datasets.
- Loss and accuracy curves are plotted across epochs to visualize model performance.

## Observations
- The CNN model consistently outperforms the simple neural network, achieving higher accuracy scores on both training and test datasets.
- Measures to combat overfitting, such as dropout, effectively mitigate the issue, ensuring generalization of the models.

## Code
The code for implementing the described functionalities can be found in the following files:
- `simple_model.py`: Contains the implementation of the simple neural network model.
- `cnn_model.py`: Contains the implementation of the CNN model.
- `data_preprocessing.py`: Includes functions for loading and preprocessing the MNIST dataset.
- `training_and_evaluation.py`: Includes code for training and evaluating both models.

## Usage
To train and evaluate the models, follow these steps:
1. Ensure all necessary dependencies are installed.
2. Run `python data_preprocessing.py` to preprocess the dataset.
3. Run `python simple_model.py` to train and evaluate the simple neural network model.
4. Run `python cnn_model.py` to train and evaluate the CNN model.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
