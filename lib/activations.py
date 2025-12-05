import numpy as np
from lib.layers import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.input > 0)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        s = 1 / (1 + np.exp(-input))
        self.output = s
        return s

    def backward(self, output_gradient):
        s = self.output
        return np.multiply(output_gradient, s * (1 - s))

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient):
        return np.multiply(output_gradient, 1 - np.power(self.output, 2))
