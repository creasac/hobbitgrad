from hobbitgrad.array import NDArray
from hobbitgrad.sgd import SGD
from hobbitgrad.tensor import Tensor


def test_step_updates_params_using_grad_and_lr():
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([10, 20])
    weight.grad = NDArray([[0.5, 1.0], [1.5, 2.0]])
    bias.grad = NDArray([3, 4])
    optimizer = SGD([weight, bias], lr=0.1)

    optimizer.step()

    assert weight.data.data == [0.95, 1.9, 2.85, 3.8]
    assert bias.data.data == [9.7, 19.6]


def test_zero_grad_resets_param_grads_to_matching_zero_arrays():
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([10, 20])
    weight.grad = NDArray([[0.5, 1.0], [1.5, 2.0]])
    bias.grad = NDArray([3, 4])
    optimizer = SGD([weight, bias], lr=0.1)

    optimizer.zero_grad()

    assert weight.grad.shape == (2, 2)
    assert weight.grad.data == [0, 0, 0, 0]
    assert bias.grad.shape == (2,)
    assert bias.grad.data == [0, 0]


def test_step_updates_param_after_backward():
    tensor = Tensor([[1, 2], [3, 4]])
    loss = (tensor * tensor).sum()
    optimizer = SGD([tensor], lr=0.1)

    loss.backward()
    optimizer.step()

    assert tensor.grad.data == [2, 4, 6, 8]
    assert tensor.data.data == [0.8, 1.6, 2.4, 3.2]
