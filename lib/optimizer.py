import numpy as np

class SGD: 
    def __init__(self, learning_rate):
        
        self.learning_rate = learning_rate
    
    def step(self, layer, batch_size=1):
        
        if hasattr(layer, 'grad_weights') and hasattr(layer, 'grad_bias'):
            if layer.grad_weights is not None:
                # Normalize gradients by batch size for proper mini-batch training
                layer.weights -= self.learning_rate * (layer.grad_weights / batch_size)
                layer.bias -= self.learning_rate * (layer.grad_bias / batch_size)
