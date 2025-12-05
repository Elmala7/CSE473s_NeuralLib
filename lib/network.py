import numpy as np
from lib.optimizer import SGD
from lib.layers import Layer

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        
        if not isinstance(layer, Layer):
            raise TypeError(f"Layer must inherit from Layer base class. Got {type(layer).__name__}")
        self.layers.append(layer)

    def use_loss(self, loss_obj):
        self.loss = loss_obj

    def predict(self, input_data):
        
        # Handle single sample
        if isinstance(input_data, list):
            samples = len(input_data)
            result = []
            for i in range(samples):
                output = np.array(input_data[i]).reshape(1, -1)
                for layer in self.layers:
                    output = layer.forward(output)
                result.append(output)
            return result
        else:
            # Handle batch (numpy array)
            input_data = np.array(input_data)
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            output = input_data
            for layer in self.layers:
                output = layer.forward(output)
            return output

    def train(self, x_train, y_train, epochs, learning_rate):
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        # Validation
        if len(x_train) != len(y_train):
            raise ValueError(f"x_train and y_train must have same length. Got {len(x_train)} and {len(y_train)}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive. Got {epochs}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive. Got {learning_rate}")
        
        samples = len(x_train)
        optimizer = SGD(learning_rate)

        for i in range(epochs):
            # Reset gradients at start of epoch
            for layer in self.layers:
                if hasattr(layer, 'grad_weights'):
                    layer.grad_weights = None
                    layer.grad_bias = None
            
            err = 0
            # Process samples one at a time (can be optimized to batch processing in future)
            for j in range(samples):
                # Forward pass
                output = np.array(x_train[j]).reshape(1, -1)
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                y_sample = np.array(y_train[j]).reshape(1, -1)
                err += self.loss.loss(y_sample, output)

                # Backward pass
                error = self.loss.loss_prime(y_sample, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)

            # Apply optimizer to update weights with batch size normalization
            for layer in self.layers:
                optimizer.step(layer, batch_size=samples)

            # Calculate average error
            err /= samples
            if i % 100 == 0:
                print(f"Epoch {i+1}/{epochs}   error={err}")
