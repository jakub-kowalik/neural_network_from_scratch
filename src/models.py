from abc import ABC, abstractmethod
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from tqdm import tqdm


class Model(ABC):
    @abstractmethod
    def __init__(self, epochs, learning_rate, loss_function, batch_size=1, verbose=True, shuffle=True):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        # if batch_size > 1:
        #     print("Batch size is not supported yet")
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle

    @abstractmethod
    def predict(self, x):
        return NotImplemented

    @abstractmethod
    def train(self, x, y):
        return NotImplemented


class Sequential(Model):
    def __init__(self, epochs, learning_rate, loss_function, batch_size=1, verbose=True, shuffle=True):
        super().__init__(epochs, learning_rate, loss_function, batch_size, verbose, shuffle)
        self.layers = []

    def add(self, layer):
        # @TODO add automatic input size detection
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_layer(self, index):
        return self.layers[index]

    def train(self, x, y):
        # @TODO later change params from model to this function
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        for epoch in (pbar := tqdm(range(self.epochs), disable=not self.verbose)):
            # @TODO add own implementation
            if self.shuffle:
                x, y = sk_shuffle(x, y)

            for i in range(0, len(x), self.batch_size):
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                output = self.predict(x_batch)

                self.loss_function.forward(output, y_batch)
                error = self.loss_function.backward()

                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            # if epoch % 100 == 0:
            pbar.set_postfix_str(f" Loss: {self.loss_function.forward(self.predict(x), y) / len(x)}")
