import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient):
        # TODO: return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Validate input
        if input_size <= 0 or output_size <= 0:
            raise ValueError(f"Layer dimensions must be positive. Got input_size={input_size}, output_size={output_size}")
        
        # He initialization for better convergence
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Accumulate gradients (add to existing, don't overwrite)
        if self.grad_weights is None:
            self.grad_weights = weights_gradient
            self.grad_bias = bias_gradient
        else:
            self.grad_weights += weights_gradient
            self.grad_bias += bias_gradient
        
        return input_gradient
