from hobbitgrad.loss import mse
from hobbitgrad.tensor import Tensor


def test_mse_returns_mean_squared_error():
    predictions = Tensor([1, 2, 4])
    target = Tensor([1, 4, 1])

    loss = mse(predictions, target)

    assert loss.data.shape == (1,)
    assert loss.data.data == [13 / 3]


def test_backward_through_mse_sets_prediction_and_target_grads():
    predictions = Tensor([1, 2, 4])
    target = Tensor([1, 4, 1])
    loss = mse(predictions, target)

    loss.backward()

    assert predictions.grad.data == [0.0, -4 / 3, 2.0]
    assert target.grad.data == [0.0, 4 / 3, -2.0]
