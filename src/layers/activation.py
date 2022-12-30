from src.activation_functions.activation_base import ActivationFunction
from src.layers.layer_base import Layer


class Activation(Layer):
    def __init__(
            self,
            activation_function: ActivationFunction,
            n_inputs: int = None,
            n_outputs: int = None
    ):
        self.activation_function = activation_function

        self.n_inputs = n_inputs
        self.n_outputs = self.n_inputs

    def initialize_weights(self):
        # No weights in activation layer
        pass

    def forward(self, x):
        self._input = x
        return self.activation_function.forward(x)

    def backward(self, error, learning_rate):
        return error * self.activation_function.backward(self._input)

    def set_input(self, x):
        self.input_size = x
        self.output_size = x
        self.n_inputs = x
        self.n_outputs = x

    def set_n_outputs(self, x):
        self.n_outputs = x

    def __str__(self):
        return self.__class__.__name__ + f"({self.activation_function})"

    def __repr__(self):
        return self.__str__()
