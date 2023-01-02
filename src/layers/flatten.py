from .layer_base import Layer
from functools import reduce


# github copilot completion
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.saved_shape = None

    def initialize_weights(self):
        # No weights in flatten layer
        pass

    def forward(self, x):
        self.saved_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, error, learning_rate):
        return error.reshape(self.saved_shape)

    def set_input(self, x):
        self.input_size = x
        self.output_size = reduce(lambda z, y: z * y, x)

    def set_n_outputs(self, x):
        pass

    def __str__(self):
        return self.__class__.__name__ \
            + "(input size: " + str(self.input_size) \
            + ", outputs: " + str(self.output_size) \
            + ")"

    def __repr__(self):
        return self.__str__()
