from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def __init__(self, epochs, learning_rate, loss_function, batch_size=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        if batch_size > 1:
            print("Batch size is not supported yet")
        self.batch_size = 1

    @abstractmethod
    def predict(self, x):
        return NotImplemented

    @abstractmethod
    def train(self, x, y):
        return NotImplemented


class Sequential(Model):
    def __init__(self, epochs, learning_rate, loss_function, batch_size=1):
        super().__init__(epochs, learning_rate, loss_function, batch_size)
        self.layers = []

    def add(self, layer):
        # @TODO add automatic input size detection
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x, y):
        for epoch in range(self.epochs):
            for i in range(0, len(x), self.batch_size):
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                output = self.predict(x_batch)

                self.loss_function.forward(output, y_batch)
                error = self.loss_function.backward()

                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            # if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {self.loss_function.forward(self.predict(x), y)}")
