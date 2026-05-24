import pytest

from hobbitgrad.array import NDArray


def test_flatten_1d():
    arr = NDArray([1, 2, 3])

    assert arr.data == [1, 2, 3]
    assert arr.shape == (3,)


def test_flatten_2d():
    arr = NDArray([[1, 2], [3, 4]])

    assert arr.data == [1, 2, 3, 4]
    assert arr.shape == (2, 2)


def test_flatten_3d():
    arr = NDArray([
        [[1], [2]],
        [[3], [4]],
    ])

    assert arr.data == [1, 2, 3, 4]
    assert arr.shape == (2, 2, 1)


def test_scalar_input():
    arr = NDArray(5)

    assert arr.data == [5]
    assert arr.shape == ()


def test_empty_list():
    arr = NDArray([])

    assert arr.data == []
    assert arr.shape == (0,)


def test_nested_empty_list():
    arr = NDArray([[], []])

    assert arr.data == []
    assert arr.shape == (2, 0)


def test_zeros_creates_array_with_shape():
    arr = NDArray.zeros((2, 3))

    assert arr.shape == (2, 3)
    assert arr.data == [0, 0, 0, 0, 0, 0]


def test_ones_creates_array_with_shape():
    arr = NDArray.ones((2, 3))

    assert arr.shape == (2, 3)
    assert arr.data == [1, 1, 1, 1, 1, 1]


def test_mixed_types():
    arr = NDArray([[1, "a"], [3.5, True]])

    assert arr.data == [1, "a", 3.5, True]
    assert arr.shape == (2, 2)


def test_getitem_1d():
    arr = NDArray([10, 20, 30])

    assert arr[(0,)] == 10
    assert arr[(1,)] == 20
    assert arr[(2,)] == 30


def test_getitem_2d():
    arr = NDArray([[1, 2], [3, 4]])

    assert arr[(0, 0)] == 1
    assert arr[(0, 1)] == 2
    assert arr[(1, 0)] == 3
    assert arr[(1, 1)] == 4


def test_getitem_3d():
    arr = NDArray([
        [[1], [2]],
        [[3], [4]],
    ])

    assert arr[(0, 0, 0)] == 1
    assert arr[(0, 1, 0)] == 2
    assert arr[(1, 0, 0)] == 3
    assert arr[(1, 1, 0)] == 4


def test_setitem_1d():
    arr = NDArray([10, 20, 30])

    arr[1] = 99

    assert arr[1] == 99
    assert arr.data == [10, 99, 30]


def test_setitem_2d():
    arr = NDArray([[1, 2], [3, 4]])

    arr[(1, 0)] = 99

    assert arr[(1, 0)] == 99
    assert arr.data == [1, 2, 99, 4]


def test_setitem_3d():
    arr = NDArray([
        [[1], [2]],
        [[3], [4]],
    ])

    arr[(0, 1, 0)] = 99

    assert arr[(0, 1, 0)] == 99
    assert arr.data == [1, 99, 3, 4]


def test_total_elements():
    arr = NDArray([[1, 2], [3, 4]])

    assert arr._total_elements((2, 2)) == 4
    assert arr._total_elements((4, 1)) == 4
    assert arr._total_elements(()) == 1


def test_is_contiguous():
    arr = NDArray([[1, 2], [3, 4]])

    assert arr._is_contiguous() is True


def test_contiguous_data():
    arr = NDArray([[1, 2], [3, 4]])

    assert arr._contiguous_data() == [1, 2, 3, 4]


def test_reshape_1d_to_2d():
    arr = NDArray([1, 2, 3, 4])

    reshaped = arr.__reshape__((2, 2))

    assert reshaped.shape == (2, 2)
    assert reshaped.data == [1, 2, 3, 4]
    assert reshaped[(0, 0)] == 1
    assert reshaped[(0, 1)] == 2
    assert reshaped[(1, 0)] == 3
    assert reshaped[(1, 1)] == 4


def test_reshape_2d_to_1d():
    arr = NDArray([[1, 2], [3, 4]])

    reshaped = arr.__reshape__((4,))

    assert reshaped.shape == (4,)
    assert reshaped.data == [1, 2, 3, 4]
    assert reshaped[0] == 1
    assert reshaped[3] == 4


def test_reshape_shape_mismatch():
    arr = NDArray([1, 2, 3, 4])

    with pytest.raises(AssertionError, match="Shape mismatch"):
        arr.__reshape__((3, 2))


def test_transpose_2d():
    arr = NDArray([[1, 2, 3], [4, 5, 6]])

    transposed = arr.__transpose__()

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert transposed[(0, 0)] == 1
    assert transposed[(0, 1)] == 4
    assert transposed[(1, 0)] == 2
    assert transposed[(2, 1)] == 6


def test_transpose_3d():
    arr = NDArray([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])

    transposed = arr.__transpose__()

    assert transposed.shape == (2, 2, 2)
    assert transposed.strides == (1, 2, 4)
    assert transposed[(0, 0, 0)] == 1
    assert transposed[(1, 0, 0)] == 2
    assert transposed[(0, 1, 0)] == 3
    assert transposed[(1, 1, 1)] == 8


def test_transpose_setitem_updates_original_data():
    arr = NDArray([[1, 2], [3, 4]])
    transposed = arr.__transpose__()

    transposed[(0, 1)] = 99

    assert transposed[(0, 1)] == 99
    assert arr[(1, 0)] == 99
    assert arr.data == [1, 2, 99, 4]


def test_transpose_is_not_contiguous():
    arr = NDArray([[1, 2, 3], [4, 5, 6]])

    transposed = arr.__transpose__()

    assert transposed._is_contiguous() is False
    assert transposed._contiguous_data() == [1, 4, 2, 5, 3, 6]


def test_matmul_returns_matrix_product():
    left = NDArray([[1, 2, 3], [4, 5, 6]])
    right = NDArray([[7, 8], [9, 10], [11, 12]])

    out = left @ right

    assert out.shape == (2, 2)
    assert out.data == [58, 64, 139, 154]
    assert out[(0, 0)] == 58
    assert out[(0, 1)] == 64
    assert out[(1, 0)] == 139
    assert out[(1, 1)] == 154


def test_matmul_with_transposed_operand():
    left = NDArray([[1, 2, 3], [4, 5, 6]])
    right = NDArray([[7, 9, 11], [8, 10, 12]]).__transpose__()

    out = left @ right

    assert out.shape == (2, 2)
    assert out.data == [58, 64, 139, 154]


def test_mul_returns_elementwise_product():
    left = NDArray([[1, 2], [3, 4]])
    right = NDArray([[5, 6], [7, 8]])

    out = left * right

    assert out.shape == (2, 2)
    assert out.data == [5, 12, 21, 32]


def test_mul_with_scalar_returns_scaled_array():
    arr = NDArray([[1, 2], [3, 4]])

    out = arr * 3

    assert out.shape == (2, 2)
    assert out.data == [3, 6, 9, 12]


def test_add_returns_elementwise_sum():
    left = NDArray([[1, 2], [3, 4]])
    right = NDArray([[5, 6], [7, 8]])

    out = left + right

    assert out.shape == (2, 2)
    assert out.data == [6, 8, 10, 12]


def test_add_with_scalar_returns_shifted_array():
    arr = NDArray([[1, 2], [3, 4]])

    out = arr + 10

    assert out.shape == (2, 2)
    assert out.data == [11, 12, 13, 14]


def test_matmul_rejects_non_2d_operands():
    left = NDArray([1, 2, 3])
    right = NDArray([[1], [2], [3]])

    with pytest.raises(ValueError, match="Expected Shape 2D"):
        left @ right


def test_matmul_rejects_mismatched_shapes():
    left = NDArray([[1, 2], [3, 4]])
    right = NDArray([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError, match="Shape mismatch"):
        left @ right
