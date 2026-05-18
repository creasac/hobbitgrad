from hobbitgrad.tensor import Tensor


def test_tensor_initializes_data_and_grad():
    tensor = Tensor(2)

    assert tensor.data == 2
    assert tensor.grad == 0


def test_mul_returns_tensor_with_product():
    left = Tensor(2)
    right = Tensor(3)

    out = left * right

    assert isinstance(out, Tensor)
    assert out.data == 6
    assert out.grad == 0


def test_backward_for_mul_sets_operand_grads():
    left = Tensor(2)
    right = Tensor(3)
    out = left * right

    out.backward()

    assert out.grad == 1
    assert left.grad == 3
    assert right.grad == 2


def test_backward_through_chained_mul():
    left = Tensor(2)
    right = Tensor(3)
    out = (left * right) * left

    out.backward()

    assert out.data == 12
    assert left.grad == 12
    assert right.grad == 4


def test_backward_when_tensor_is_reused_in_mul():
    tensor = Tensor(4)
    out = tensor * tensor

    out.backward()

    assert out.data == 16
    assert tensor.grad == 8
