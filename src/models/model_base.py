from abc import ABC, abstractmethod
from typing import Tuple, Union
import matplotlib.pyplot as plt


class Model(ABC):
    @abstractmethod
    def __init__(
            self,
            input_size: Union[int, Tuple[int, int, int]],
            output_size: int
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

        self.training_losses = []
        self.validation_losses = []

    @abstractmethod
    def predict(self, x, training=False):
        return NotImplemented

    @abstractmethod
    def fit(self, x, y):
        return NotImplemented

    def summary(self):
        print(str(self.__class__.__name__) + " {")
        for layer in self.layers:
            print("\t" + str(layer))
        print("}")

    def plot_history(self):
        plt.title("Model loss")
        plt.plot(self.training_losses, label="train")
        plt.plot(self.validation_losses, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
