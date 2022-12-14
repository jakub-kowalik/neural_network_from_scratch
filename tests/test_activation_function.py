from src.activation_functions import Sigmoid, ReLU, Softmax, Tanh

import tensorflow as tf
import scipy.special
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress tensorflow warnings


def test_sigmoid_forward():
    sigmoid = Sigmoid()
    random_array = np.random.rand(3, 3, 3)

    assert np.isclose(
        sigmoid.forward(random_array),
        tf.nn.sigmoid(random_array)
    ).all()


def test_sigmoid_backward():
    sigmoid = Sigmoid()
    random_array = np.random.rand(3, 3, 3)
    sigmoid.forward(random_array)

    assert np.isclose(
        sigmoid.backward(0),
        tf.nn.sigmoid(sigmoid._output) * (1 - tf.nn.sigmoid(sigmoid._output))
    ).all()


def test_relu_forward():
    relu = ReLU()
    random_array = np.random.rand(3, 3, 3)

    assert np.isclose(
        relu.forward(random_array),
        tf.nn.relu(random_array)
    ).all()


def test_relu_backward():
    relu = ReLU()
    random_array = np.random.rand(3, 3, 3)

    relu.forward(random_array)
    assert np.isclose(
        relu.backward(random_array),
        np.where(np.maximum(0, random_array) > 0, 1, 0)
    ).all()


def test_softmax_forward():
    softmax = Softmax()
    random_array = np.random.rand(3, 3, 3)

    assert np.isclose(
        softmax.forward(random_array),
        tf.nn.softmax(random_array, axis=1)  # axis 1 because of batching
    ).all()


def test_softmax_backward():
    # should just return 1 in current implementation
    softmax = Softmax()
    random_array = np.random.rand(3, 3, 3)
    softmax.forward(random_array)

    assert np.isclose(
        softmax.backward(random_array),
        1
    ).all()


def test_tanh_forward():
    tanh = Tanh()
    random_array = np.random.rand(3, 3, 3)

    assert np.isclose(
        tanh.forward(random_array),
        tf.nn.tanh(random_array)
    ).all()


def test_tanh_backward():
    tanh = Tanh()
    random_array = np.random.rand(3, 3, 3)
    tanh.forward(random_array)

    assert np.isclose(
        tanh.backward(random_array),
        1 - tf.nn.tanh(tanh._output) ** 2
    ).all()
