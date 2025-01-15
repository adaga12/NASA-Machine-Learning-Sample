# NASA-Machine-Learning-Sample
This is a sample of the work that I am doing while working as a machine learning intern at NASA. 

This repository contains a TensorFlow-based implementation of a deep learning framework to approximate the parameters of sine functions (amplitude, frequency, and phase) from noisy synthetic data. The model is trained to predict these parameters, leveraging a fully connected neural network and custom training logic.

Features:

Synthetic Data Generation:
- Simulates noisy sine wave data with customizable amplitude, frequency, phase, time decay, and noise.
- Generates training, validation, and test datasets for model training and evaluation.

Deep Learning Model:
- Fully connected neural network with ReLU activation layers for non-linear parameter estimation.
- Outputs predictions for the sine wave parameters (amplitude, frequency, phase).

Custom Training Loop:
- Implements a weight adjustment algorithm using TensorFlow’s GradientTape for manual backpropagation and weight updates.
- Tracks loss across training epochs for both labeled and unlabeled datasets.
  
Model Evaluation:
- Compares predicted parameters against actual values on a test dataset.
- Outputs predictions and actual values to a file for analysis.
