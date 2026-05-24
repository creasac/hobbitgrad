from hobbitgrad.tensor import Tensor
from hobbitgrad.matlib import NDArray


def test_tensor_initializes_data_and_grad():
    tensor = Tensor(2)

    assert tensor.data.data[0] == 2
    assert tensor.grad.data[0] == 0


def test_tensor_initializes_list_data_and_matching_grad():
    tensor = Tensor([[1, 2], [3, 4]])

    assert tensor.data.shape == (2, 2)
    assert tensor.data.data == [1, 2, 3, 4]
    assert tensor.grad.shape == (2, 2)
    assert tensor.grad.data == [0, 0, 0, 0]


def test_tensor_initializes_ndarray_data_and_matching_grad():
    data = NDArray([[1, 2], [3, 4]])

    tensor = Tensor(data)

    assert tensor.data is data
    assert tensor.grad.shape == (2, 2)
    assert tensor.grad.data == [0, 0, 0, 0]


def test_mul_returns_tensor_with_product():
    left = Tensor(2)
    right = Tensor(3)

    out = left * right

    assert isinstance(out, Tensor)
    assert out.data.data[0] == 6
    assert out.grad.data[0] == 0


def test_add_returns_tensor_with_sum():
    left = Tensor(2)
    right = Tensor(3)

    out = left + right

    assert isinstance(out, Tensor)
    assert out.data.data[0] == 5
    assert out.grad.data[0] == 0


def test_backward_for_mul_sets_operand_grads():
    left = Tensor(2)
    right = Tensor(3)
    out = left * right

    out.backward()

    assert out.grad.data[0] == 1
    assert left.grad.data[0] == 3
    assert right.grad.data[0] == 2


def test_backward_for_add_sets_operand_grads():
    left = Tensor(2)
    right = Tensor(3)
    out = left + right

    out.backward()

    assert out.grad.data[0] == 1
    assert left.grad.data[0] == 1
    assert right.grad.data[0] == 1


def test_backward_through_chained_mul():
    left = Tensor(2)
    right = Tensor(3)
    out = (left * right) * left

    out.backward()

    assert out.data.data[0] == 12
    assert left.grad.data[0] == 12
    assert right.grad.data[0] == 4


def test_backward_through_chained_add():
    left = Tensor(2)
    middle = Tensor(3)
    right = Tensor(4)
    out = (left + middle) + right

    out.backward()

    assert out.data.data[0] == 9
    assert left.grad.data[0] == 1
    assert middle.grad.data[0] == 1
    assert right.grad.data[0] == 1


def test_backward_through_add_and_mul():
    left = Tensor(2)
    right = Tensor(3)
    out = (left + right) * left

    out.backward()

    assert out.data.data[0] == 10
    assert left.grad.data[0] == 7
    assert right.grad.data[0] == 2


def test_backward_when_tensor_is_reused_in_mul():
    tensor = Tensor(4)
    out = tensor * tensor

    out.backward()

    assert out.data.data[0] == 16
    assert tensor.grad.data[0] == 8


def test_backward_when_tensor_is_reused_in_add():
    tensor = Tensor(4)
    out = tensor + tensor

    out.backward()

    assert out.data.data[0] == 8
    assert tensor.grad.data[0] == 2


def test_matmul_returns_tensor_with_matrix_product():
    left = Tensor([[1, 2, 3], [4, 5, 6]])
    right = Tensor([[7, 8], [9, 10], [11, 12]])

    out = left @ right

    assert isinstance(out, Tensor)
    assert out.data.shape == (2, 2)
    assert out.data.data == [58, 64, 139, 154]
    assert out.grad.shape == (2, 2)
    assert out.grad.data == [0, 0, 0, 0]


def test_backward_for_matmul_sets_operand_grads():
    left = Tensor([[1, 2, 3], [4, 5, 6]])
    right = Tensor([[7, 8], [9, 10], [11, 12]])
    out = left @ right

    out.backward()

    assert out.grad.data == [1, 1, 1, 1]
    assert left.grad.shape == (2, 3)
    assert left.grad.data == [15, 19, 23, 15, 19, 23]
    assert right.grad.shape == (3, 2)
    assert right.grad.data == [5, 5, 7, 7, 9, 9]
