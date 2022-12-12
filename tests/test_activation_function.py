from src.activation_functions import Sigmoid, ReLU, Softmax
import scipy.special
import numpy as np


def test_sigmoid_forward():
    sigmoid = Sigmoid()
    random_array = np.random.rand(3, 3, 3)

    assert np.isclose(sigmoid.forward(random_array), scipy.special.expit(random_array)).all()


def test_sigmoid_backward():
    sigmoid = Sigmoid()
    random_array = np.random.rand(3, 3, 3)
    sigmoid.forward(random_array)

    assert np.isclose(sigmoid.backward(0),
                      scipy.special.expit(sigmoid._output) * (scipy.special.expit(1 - sigmoid._output))).all()


def test_relu_forward():
    relu = ReLU()
    random_array = np.random.rand(3, 3, 3)

    assert np.all(relu.forward(random_array) == np.maximum(0, random_array))


def test_relu_backward():
    relu = ReLU()
    random_array = np.random.rand(3, 3, 3)

    relu.forward(random_array)
    assert np.all(relu.backward(random_array) == np.where(np.maximum(0, random_array) > 0, 1, 0))


def test_softmax_forward():
    softmax = Softmax()
    random_array = np.random.rand(3, 3, 3)

    assert np.isclose(softmax.forward(random_array), scipy.special.softmax(random_array)).all()


# def test_softmax_backward():
#     softmax = Softmax()
#     random_array = np.random.rand(3, 3, 3)
#
#     assert np.isclose(softmax.forward(random_array), scipy.special.softmax(random_array)).all()

