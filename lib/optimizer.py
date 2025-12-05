import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer.
    
    Updates model parameters using the formula:
    W_new = W_old - learning_rate * grad_weights
    b_new = b_old - learning_rate * grad_bias
    """
    
    def __init__(self, learning_rate):
        """Initialize SGD optimizer.
        
        Args:
            learning_rate (float): Learning rate for gradient descent updates
        """
        self.learning_rate = learning_rate
    
    def step(self, layer):
        """Update weights and biases for a single layer.
        
        Args:
            layer: Layer object with grad_weights and grad_bias attributes
        """
        if hasattr(layer, 'grad_weights') and hasattr(layer, 'grad_bias'):
            if layer.grad_weights is not None:
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias
