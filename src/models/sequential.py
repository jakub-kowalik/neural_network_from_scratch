from src.loss_functions.loss_function_base import LossFunction
from src.models.model_base import Model
from typing import Union, Tuple, List
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Sequential(Model):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int, int]],
        output_size: int,
        layers: List = None,
    ):
        super().__init__(input_size, output_size)

        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if len(self.layers) == 0:
            layer.set_input(self.input_size)
        else:
            layer.set_input(self.layers[-1].output_size)

        layer.initialize_weights()

        self.layers.append(layer)

    def predict(self, x, training=False):
        if training:
            for layer in self.layers:
                x = layer.forward(x)
        else:
            for layer in self.layers:
                if type(layer).__name__ == "Dropout":
                    continue
                x = layer.forward(x)
        return x

    def get_layer(self, index):
        return self.layers[index]

    def fit(
        self,
        training_data: Union[List, Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 10,
        learning_rate: float = 0.0001,
        loss_function: LossFunction = None,
        batch_size: int = 1,
        verbose: bool = False,
        shuffle: bool = True,
        validation_data: Union[List, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        assert (
            self.output_size == self.layers[-1].output_size
        ), "Output size of model must match output size of last layer"

        x = training_data[0]
        y = training_data[1]

        for epoch in (pbar := tqdm(range(epochs), disable=not verbose)):
            # @TODO add own implementation
            if shuffle:
                x, y = sk_shuffle(x, y)

            for i in range(0, len(x), batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                output = self.predict(x_batch, training=True)

                loss_function.forward(output, y_batch)
                error = loss_function.backward()

                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            train_predictions = self.predict(x, training=True)

            train_loss = loss_function.forward(train_predictions, y) / len(x)
            train_accuracy = accuracy_score(
                np.argmax(y, axis=1), np.argmax(train_predictions, axis=1)
            )
            self.training_losses.append(train_loss)
            self.training_accuracies.append(train_accuracy)

            postfix_string = f" Train loss: {train_loss:.4f}"
            postfix_string += f" Train accuracy: {train_accuracy:.2f}"

            if validation_data is not None:
                val_predictions = self.predict(validation_data[0], training=True)
                val_loss = loss_function.forward(
                    val_predictions, validation_data[1]
                ) / len(validation_data[0])
                val_accuracy = accuracy_score(
                    np.argmax(val_predictions, axis=1),
                    np.argmax(validation_data[1], axis=1),
                )
                postfix_string += f" Val loss: {val_loss:.4f}"
                postfix_string += f" Val accuracy: {val_accuracy:.2f}"
                self.validation_losses.append(val_loss)
                self.validation_accuracies.append(val_accuracy)

            pbar.set_postfix_str(postfix_string)
