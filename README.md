# CSE473s: Neural Network Library

## Description

A NumPy-based neural network library built from scratch for educational purposes. This project implements the core components of a deep learning framework, including layers, activations, loss functions, and a training engine.

## Current Status

**Project: COMPLETE ✅**

- ✅ Part 1 (Core library & verification): Gradient checking passed (diff = 2.81e-12), XOR solved (100%).
- ✅ Part 2 (Autoencoder & Analysis): Autoencoder reconstruction MSE (test): **0.007915**; SVM on latent features accuracy: **96.89%**. Keras benchmark reconstruction MSE: **0.002647** (lower but comparable qualitative reconstructions).
- ✅ Report and notebook: see `report/project_report.md`, `report/Part1_Report.tex`, and `notebooks/project_demo.ipynb`. 

## Project Structure

```
CSE473s_NeuralLib/
├── lib/                          # Core neural network library
│   ├── layers.py                 # Dense layer implementation
│   ├── activations.py            # ReLU, Sigmoid, Tanh activations
│   ├── losses.py                 # Mean Squared Error (MSE) loss
│   ├── network.py                # Sequential model class
│   └── optimizer.py              # SGD optimizer
├── notebooks/
│   └── project_demo.ipynb        # Gradient check & XOR training
├── report/
│   └── part1_summary.md          # Part 1 results summary
├── Docs/
│   ├── CSE473s_specifications.md # Project specifications
│   └── CSE473s_todo.md           # Task tracking
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Features Implemented

### Mathematical Engine
- **Dense Layer**: Full forward and backward pass with proper gradient computation
- **Activations**: ReLU, Sigmoid, Tanh with correct derivatives
- **Loss Function**: MSE with correct gradient scaling
- **Optimizer**: Stochastic Gradient Descent (SGD)

### Validation
- Gradient checking (numerical vs. analytical gradients)
- XOR problem as benchmark task
- 100% training accuracy achieved

## Results (Summary)

| Task | Result |
|------|--------|
| Gradient Check | ✅ PASSED (diff = 2.81e-12) |
| XOR Accuracy | ✅ 100% |
| Final Loss (XOR) | 0.000030051 |
| Autoencoder (reconstruction MSE, test) | 0.007915 |
| Autoencoder (Keras reconstruction MSE, test) | 0.002647 |
| SVM on Latent Features (Accuracy) | 96.89% |
| Training Epochs (XOR) | 10,000 |
| Best Learning Rate | 1.0 |
| Weight Initialization | He initialization |

## Completed Work (Part 2)

- ✅ Autoencoder on MNIST (784 -> 128 -> 784). Reconstruction MSE (test): **0.007915**
- ✅ SVM classification on latent vectors (Accuracy: **96.89%**)
- ✅ Keras/TensorFlow benchmark completed (Keras MSE: **0.002647**)
- ✅ Full report and notebook available for reproduction

## Requirements

- Python 3.10+
- NumPy
- Jupyter Notebook (for experiments)

## Usage

```python
from lib.layers import Dense
from lib.activations import Tanh, Sigmoid
from lib.losses import MSE
from lib.network import Sequential
from lib.optimizer import SGD
import numpy as np

# Create model
model = Sequential()
model.add(Dense(2, 16))
model.add(Tanh())
model.add(Dense(16, 1))
model.add(Sigmoid())
model.use_loss(MSE())

# Train
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
model.train(X, y, epochs=10000, learning_rate=1.0)

# Predict
predictions = model.predict(X)
```

## Author

Zaid Reda
