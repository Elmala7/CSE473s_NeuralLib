import numpy as np

class SGD: 
    def __init__(self, learning_rate):
        
        self.learning_rate = learning_rate
    
    def step(self, layer):

        if hasattr(layer, 'grad_weights') and hasattr(layer, 'grad_bias'):
            if layer.grad_weights is not None:
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias
