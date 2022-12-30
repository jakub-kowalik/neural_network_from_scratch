from src.activation_functions.activation_base import ActivationFunction


class ReLU(ActivationFunction):
    # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    def forward(self, x):
        self._output = x * (x > 0)
        return self._output

    def backward(self, x):
        self._error = 1. * (self._output > 0)
        return self._error
