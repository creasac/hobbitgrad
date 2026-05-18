from matlib import NDArray


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
