# CSE473s: Neural Network Library

![Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-Required-orange)

## 📌 Overview

A NumPy-based neural network library built from scratch for educational purposes. This project implements the core components of a deep learning framework, including dense layers, non-linear activations, loss functions, and an optimizer engine.

## ✨ Features Implemented

### Mathematical Engine
- **Dense Layer**: Full forward and backward pass with proper gradient computation.
- **Activations**: ReLU, Sigmoid, and Tanh with analytically derived gradients.
- **Loss Function**: Mean Squared Error (MSE) with correct gradient scaling.
- **Optimizer**: Stochastic Gradient Descent (SGD).

### Validation & Testing
- **Gradient Checking**: Verifies analytical gradients against numerical approximations.
- **Benchmarks**: Successfully solves the XOR problem and trains an Autoencoder on the MNIST dataset.

## 🚀 Quick Start

### Requirements
- Python 3.10+
- NumPy
- Jupyter Notebook (optional, for running experiments)

### Example Usage

```python
import numpy as np
from lib.network import Sequential
from lib.layers import Dense
from lib.activations import Tanh, Sigmoid
from lib.losses import MSE

# 1. Define Model Architecture
model = Sequential()
model.add(Dense(2, 16))
model.add(Tanh())
model.add(Dense(16, 1))
model.add(Sigmoid())

# 2. Configure Loss Function
model.use_loss(MSE())

# 3. Prepare Data (XOR Problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 4. Train
model.train(X, y, epochs=10000, learning_rate=1.0)

# 5. Predict
predictions = model.predict(X)
print(predictions)
```

## 📊 Results & Benchmarks

The library has been rigorously tested against standard benchmarks and compared with Keras/TensorFlow implementations.

| Metric / Task | Result |
|---------------|--------|
| **Gradient Check (Numerical vs Analytical)** | ✅ PASSED (diff = 2.81e-12) |
| **XOR Problem Accuracy** | ✅ 100% |
| **Final Loss (XOR)** | 0.000030051 |
| **Autoencoder Reconstruction MSE (Custom Lib)** | 0.007915 |
| **Autoencoder Reconstruction MSE (Keras)** | 0.002647 |
| **SVM on Latent Features Accuracy** | 96.89% |

*Note: While Keras achieves a lower MSE, the qualitative reconstructions from our custom library are highly comparable. Detailed analyses can be found in the `report/` and `notebooks/` directories.*

## 📁 Project Structure

```text
CSE473s_NeuralLib/
├── lib/                          # Core neural network library
│   ├── layers.py                 # Dense layer implementation
│   ├── activations.py            # ReLU, Sigmoid, Tanh activations
│   ├── losses.py                 # Mean Squared Error (MSE) loss
│   ├── network.py                # Sequential model class
│   └── optimizer.py              # SGD optimizer
├── notebooks/                    # Jupyter notebooks for experiments
│   └── project_demo.ipynb        # Gradient check, XOR, and Autoencoder demo
├── report/                       # Project reports and documentation
│   ├── part1_summary.md          
│   ├── project_report.md
│   └── Part1_Report.tex
├── Docs/                         # Specifications and tasks
│   ├── CSE473s_specifications.md 
│   └── CSE473s_todo.md           
└── README.md                     # This documentation
```

## 👨‍💻 Author

**Zaid Reda**
