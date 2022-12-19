import numpy as np

from src.layers import Flatten, Conv2DVectorized

import tensorflow as tf


def test_flatten():
    flat = Flatten()
    random_array = np.random.random((3, 4, 5))

    forward_output = flat.forward(random_array)
    backward_output = flat.backward(forward_output, 0)

    assert forward_output.shape == (3, 20)
    assert backward_output.shape == random_array.shape
    assert np.isclose(random_array, backward_output).all()


def test_conv2d_batch_controlled():
    the_image = np.asarray([
        [[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]],
        [[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]],
        [[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]]
    ])

    filter_weights = np.asarray([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    my_conv = Conv2DVectorized(
        n_outputs=1,
        filter_size=3,
        stride=1,
        padding=1
    )

    my_conv.weights = np.expand_dims(filter_weights, 0)

    tf_conv2d = tf.nn.conv2d(
        input=np.einsum("bchw -> bhwc", the_image),
        filters=np.expand_dims(filter_weights, (2, 3)),
        padding='SAME',
        data_format='NHWC',
        strides=1
    )

    assert np.isclose(
        my_conv.forward(the_image),
        np.einsum("NHWC->NCHW", tf_conv2d)
    ).all()


import numpy as np

from src.layers import Flatten, Conv2DVectorized

import tensorflow as tf


def test_flatten():
    flat = Flatten()
    random_array = np.random.random((3, 4, 5))

    forward_output = flat.forward(random_array)
    backward_output = flat.backward(forward_output, 0)

    assert forward_output.shape == (3, 20)
    assert backward_output.shape == random_array.shape
    assert np.isclose(random_array, backward_output).all()


def test_conv2d_batch_random():
    batch = np.random.randint(1, 100)
    channels = 1  # @TODO figure out how tensorflow handles multiple channels in weights
    height = np.random.randint(10, 100)
    width = height
    the_image = np.random.random((batch, channels, height, width)).astype(np.float32)

    print(the_image.shape)

    filter_size = np.random.randint(1, 10)
    filter_weights = np.random.random((
        filter_size,
        filter_size
    ))

    my_conv = Conv2DVectorized(
        n_outputs=1,
        filter_size=filter_size,
        stride=1,
        padding=0
    )

    my_conv.weights = np.expand_dims(filter_weights, 0).astype(np.float32)

    tf_conv2d = tf.nn.conv2d(
        input=np.einsum("bchw -> bhwc", the_image),
        filters=np.expand_dims(filter_weights, (2, 3)),
        padding='VALID',
        data_format='NHWC',
        strides=1
    )
    print(my_conv.forward(the_image).shape)
    assert np.isclose(
        my_conv.forward(the_image),
        np.einsum("NHWC->NCHW", tf_conv2d)
    ).all()



