# CSE473s: Neural Network Library

## Description

A NumPy-based neural network library built from scratch for educational purposes. This project implements the core components of a deep learning framework, including layers, activations, loss functions, and a training engine.

## Current Status

**Part 1: Core Library & Validation ✅ COMPLETE**

- ✅ Gradient checking verified (difference = 2.81e-12)
- ✅ XOR problem solved with 100% accuracy
- ✅ Final loss: 0.00003
- ✅ Predictions: [0, 1, 1, 0] (perfect match)

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

## Results (Part 1)

| Task | Result |
|------|--------|
| Gradient Check | ✅ PASSED (diff = 2.81e-12) |
| XOR Accuracy | ✅ 100% |
| Final Loss | 0.000030051 |
| Training Epochs | 10,000 |
| Best Learning Rate | 1.0 |
| Weight Initialization | He initialization |

## Future Work (Part 2)

- [ ] Autoencoder on MNIST
- [ ] SVM classification on latent vectors
- [ ] Keras/TensorFlow benchmark
- [ ] Comprehensive performance report

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

## License

MIT
