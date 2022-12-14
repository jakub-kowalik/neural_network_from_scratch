import numpy as np

from src.layers import Flatten


def test_flatten():
    flat = Flatten()
    random_array = np.random.random((3, 4, 5))

    forward_output = flat.forward(random_array)
    backward_output = flat.backward(forward_output, 0)

    assert forward_output.shape == (3, 20)
    assert backward_output.shape == random_array.shape
    assert np.isclose(random_array, backward_output).all()
