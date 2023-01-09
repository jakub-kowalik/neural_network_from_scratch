import numpy as np
import torch

from src.layers import Flatten, Conv2D


def test_conv2d_batch_controlled():
    the_image = np.asarray(
        [
            [
                [
                    [2, 2, 0, 0, 0],
                    [2, 0, 1, 0, 1],
                    [2, 0, 1, 2, 1],
                    [2, 0, 0, 1, 0],
                    [0, 2, 1, 0, 1],
                ],
                [
                    [2, 1, 1, 2, 0],
                    [0, 2, 1, 2, 1],
                    [2, 2, 1, 0, 2],
                    [2, 2, 2, 2, 0],
                    [1, 2, 0, 2, 1],
                ],
                [
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 2, 0],
                    [2, 0, 1, 1, 2],
                    [1, 1, 2, 1, 2],
                    [0, 2, 1, 0, 1],
                ],
            ],
            [
                [
                    [2, 2, 0, 0, 0],
                    [2, 0, 1, 0, 1],
                    [2, 0, 1, 2, 1],
                    [2, 0, 0, 1, 0],
                    [0, 2, 1, 0, 1],
                ],
                [
                    [2, 1, 1, 2, 0],
                    [0, 2, 1, 2, 1],
                    [2, 2, 1, 0, 2],
                    [2, 2, 2, 2, 0],
                    [1, 2, 0, 2, 1],
                ],
                [
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 2, 0],
                    [2, 0, 1, 1, 2],
                    [1, 1, 2, 1, 2],
                    [0, 2, 1, 0, 1],
                ],
            ],
        ]
    )

    weights = np.asarray(
        [
            [
                [[1, 0, 0], [0, 0, -1], [0, 0, -1]],
                [[1, 1, 0], [-1, 1, -1], [1, 1, 1]],
                [[0, -1, 1], [1, -1, -1], [0, -1, 1]],
            ],
            [
                [[0, -1, 1], [0, 0, -1], [1, 0, -1]],
                [[1, 1, -1], [0, 0, 0], [-1, 0, -1]],
                [[0, 0, 1], [0, 0, -1], [0, 1, -1]],
            ],
        ]
    )

    pytorch_conv = torch.nn.Conv2d(
        in_channels=3,
        out_channels=2,
        kernel_size=3,
        bias=False,
        stride=1,
        padding_mode="zeros",
        padding=0,
    )

    x_tensor = torch.from_numpy(the_image.astype(np.float32))
    x_tensor.requires_grad = True
    pytorch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(weights.astype(np.float32))
    )

    out_torch = pytorch_conv.forward(x_tensor).detach().numpy()

    my_conv = Conv2D(
        input_size=(3, 5, 5), n_outputs=2, filter_size=3, stride=1, padding=0
    )

    my_conv.weights = np.einsum("abcd -> bacd", weights)

    out_conv2d = my_conv.forward(the_image)

    assert np.isclose(out_conv2d, out_torch).all()


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
    channels = np.random.randint(1, 10)
    height = np.random.randint(10, 100)
    width = height
    the_image = np.random.random((batch, channels, height, width)).astype(np.float32)

    n_outputs = np.random.randint(1, 10)

    weights = np.random.random((n_outputs, channels, 3, 3)).astype(np.float32)

    pytorch_conv = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=n_outputs,
        kernel_size=3,
        bias=False,
        stride=1,
        padding_mode="zeros",
        padding=0,
    )

    x_tensor = torch.from_numpy(the_image.astype(np.float32))
    x_tensor.requires_grad = True
    pytorch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(weights.astype(np.float32))
    )

    out_torch = pytorch_conv.forward(x_tensor).detach().numpy()

    my_conv = Conv2D(
        input_size=(channels, height, width),
        n_outputs=n_outputs,
        filter_size=3,
        stride=1,
        padding=0,
    )

    my_conv.weights = np.einsum("abcd -> bacd", weights)

    out_conv2d = my_conv.forward(the_image)

    assert np.isclose(out_conv2d, out_torch).all()
