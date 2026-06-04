from hobbitgrad.tensor import Tensor
from hobbitgrad.array import NDArray


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


def test_sum_returns_tensor_with_scalar_total():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    out = tensor.sum()

    assert isinstance(out, Tensor)
    assert out.data.shape == (1,)
    assert out.data.data == [21]
    assert out.grad.shape == (1,)
    assert out.grad.data == [0]


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


def test_backward_for_sum_sets_input_grad_to_ones():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    out = tensor.sum()

    out.backward()

    assert out.grad.data == [1]
    assert tensor.grad.shape == (2, 3)
    assert tensor.grad.data == [1, 1, 1, 1, 1, 1]


def test_backward_through_sum_after_mul_reduces_loss_to_scalar():
    left = Tensor([[1, 2], [3, 4]])
    right = Tensor([[5, 6], [7, 8]])
    loss = (left * right).sum()

    loss.backward()

    assert loss.data.data == [70]
    assert left.grad.shape == (2, 2)
    assert left.grad.data == [5, 6, 7, 8]
    assert right.grad.shape == (2, 2)
    assert right.grad.data == [1, 2, 3, 4]


def test_backward_for_add_sums_broadcasted_row_vector_grad():
    left = Tensor([[1, 2, 3], [4, 5, 6]])
    right = Tensor([10, 20, 30])
    out = left + right

    out.backward()

    assert out.data.shape == (2, 3)
    assert left.grad.shape == (2, 3)
    assert left.grad.data == [1, 1, 1, 1, 1, 1]
    assert right.grad.shape == (3,)
    assert right.grad.data == [2, 2, 2]


def test_backward_for_add_sums_broadcasted_column_vector_grad():
    left = Tensor([[1, 2, 3], [4, 5, 6]])
    right = Tensor([[10], [20]])
    out = left + right

    out.backward()

    assert out.data.shape == (2, 3)
    assert left.grad.shape == (2, 3)
    assert left.grad.data == [1, 1, 1, 1, 1, 1]
    assert right.grad.shape == (2, 1)
    assert right.grad.data == [3, 3]


def test_backward_for_add_sums_left_operand_broadcast_grad():
    left = Tensor([10, 20, 30])
    right = Tensor([[1, 2, 3], [4, 5, 6]])
    out = left + right

    out.backward()

    assert out.data.shape == (2, 3)
    assert left.grad.shape == (3,)
    assert left.grad.data == [2, 2, 2]
    assert right.grad.shape == (2, 3)
    assert right.grad.data == [1, 1, 1, 1, 1, 1]


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
