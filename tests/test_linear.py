from hobbitgrad.linear import Linear
from hobbitgrad.tensor import Tensor


def test_forward_returns_affine_transform_with_bias():
    layer = Linear(3, 2)
    layer.W = Tensor([[1, 2], [3, 4], [5, 6]])
    layer.b = Tensor([10, 20])
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    out = layer.forward(x)

    assert out.data.shape == (2, 2)
    assert out.data.data == [32, 48, 59, 84]


def test_parameters_returns_weight_and_bias():
    layer = Linear(3, 2)

    params = layer.parameters()

    assert params == [layer.W, layer.b]


def test_backward_through_linear_sum_sets_parameter_grads():
    layer = Linear(3, 2)
    layer.W = Tensor([[1, 2], [3, 4], [5, 6]])
    layer.b = Tensor([10, 20])
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    loss = layer.forward(x).sum()

    loss.backward()

    assert loss.data.data == [223]
    assert x.grad.shape == (2, 3)
    assert x.grad.data == [3, 7, 11, 3, 7, 11]
    assert layer.W.grad.shape == (3, 2)
    assert layer.W.grad.data == [5, 5, 7, 7, 9, 9]
    assert layer.b.grad.shape == (2,)
    assert layer.b.grad.data == [2, 2]
