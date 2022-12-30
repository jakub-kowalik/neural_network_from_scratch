from src.layers.layer_base import Layer
import numpy as np


# github copilot completion
class Dropout(Layer):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.mask = None

    def initialize_weights(self):
        # No weights in dropout layer
        pass

    def forward(self, x):
        self.mask = np.random.binomial(1, self.p, size=x.shape)
        return x * self.mask

    def backward(self, error, learning_rate):
        return error * self.mask

    def set_input(self, x):
        self.input_size = x
        self.output_size = x

    def set_n_outputs(self, x):
        pass

    def __str__(self):
        return self.__class__.__name__ + f"({self.p})"

    def __repr__(self):
        return self.__str__()
