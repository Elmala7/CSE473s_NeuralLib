import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use_loss(self, loss_obj):
        self.loss = loss_obj

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = np.array(input_data[i]).reshape(1, -1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
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
                    error = layer.backward(error, learning_rate)

            # Calculate average error
            err /= samples
            if i % 100 == 0:
                print(f"Epoch {i+1}/{epochs}   error={err}")
