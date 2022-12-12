import numpy as np

from src.layers import Linear
from src.loss_functions import TestError
from src.models import Sequential


def test_model_1_3_1():
    lin1 = Linear(n_inputs=1, n_outputs=3)
    lin1.weights = np.asarray([[0.1, -0.1, 0.3]])

    out1 = Linear(n_inputs=3, n_outputs=1)
    out1.weights = np.asarray([[0.7], [0.9], [-0.4]])

    model = Sequential(epochs=1, learning_rate=0.01, loss_function=TestError())

    model.add(lin1)
    model.add(out1)

    model.train(np.asarray([[0.5]]), np.asarray([[0.1]]))

    assert np.isclose(model.layers[0].weights,
                      np.asarray([[0.100875, -0.1, 0.2995]]), rtol=1e-1).all()

    assert np.isclose(model.layers[1].weights,
                      np.asarray([[0.700125], [0.9], [-0.399625]]), rtol=1e-1).all()


def test_model_1_1_1():
    lin1 = Linear(n_inputs=1, n_outputs=1)
    lin1.weights = np.asarray([[0.1]]).astype('float32')

    out1 = Linear(n_inputs=1, n_outputs=1)
    out1.weights = np.asarray([[0.3]]).astype('float32')

    model = Sequential(epochs=1, learning_rate=0.01, loss_function=TestError())

    model.add(lin1)
    model.add(out1)

    model.train(np.asarray([[0.5]]), np.asarray([[0.1]]))

    assert np.isclose(model.layers[0].weights,
                      np.asarray([[0.100255]])).all()

    assert np.isclose(model.layers[1].weights,
                      np.asarray([[0.300085]])).all()
