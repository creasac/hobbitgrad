import itertools
import random

class NDArray:
    def __init__(self, data, shape=None, strides=None):
        if shape is not None:
            self.data = data
            self.shape = shape
            self.strides = strides if strides is not None else self._compute_strides(shape)
        else:
            self.data, self.shape = self._flatten(data)
            self.strides = self._compute_strides(self.shape)

    @classmethod
    def zeros(cls, shape):
        def make_nested(shape):
            if len(shape) == 1:
                return [0] * shape[0]
            return [make_nested(shape[1:]) for _ in range(shape[0])]
        return cls(make_nested(shape))
    
    @classmethod
    def ones(cls, shape):
        def make_nested(shape):
            if len(shape) == 1:
                return [1] * shape[0]
            return [make_nested(shape[1:]) for _ in range(shape[0])]
        return cls(make_nested(shape))
    
    @classmethod
    def randn(cls, shape):
        def make_nested(shape):
            if len(shape) == 1:
                return [random.gauss(0, 1) for _ in range(shape[0])]
            return [make_nested(shape[1:]) for _ in range(shape[0])]
        return cls(make_nested(shape))

    def _flatten(self, data):
        # recursively go down the nested list to infer shape
        shape = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            if len(current) == 0:
                break
            lengths = set(len(x) for x in current if isinstance(x, list))
            if len(lengths) > 1:
                raise ValueError(f"Jagged array detected at depth {len(shape)}")
            current = current[0]

        flat = []
        def _recurse(x):
            if isinstance(x, list):
                for item in x:
                    _recurse(item)
            else:
                flat.append(x)
        _recurse(data)
        return flat, tuple(shape)
    
    def _compute_strides(self, shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
    
    def _flat_index(self, indices):
        return sum(i * s for i, s in zip(indices, self.strides))
    
    def _total_elements(self, shape):
        result = 1
        for s in shape:
            result *= s
        return result
    
    def _is_contiguous(self):
        expected = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.strides[i] != expected:
                return False
            expected *= self.shape[i]
        return True
    
    def _contiguous_data(self):
        # iterate over all indices to order elements logically
        ranges = [range(s) for s in self.shape]
        return [self.data[self._flat_index(idx)] for idx in itertools.product(*ranges)]
    
    def reshape(self, new_shape):
        assert self._total_elements(new_shape) == len(self.data), "Shape mismatch"
        data = self._contiguous_data() if not self._is_contiguous() else self.data
        return NDArray(data, new_shape)
    
    def transpose(self):
        return NDArray(self.data, tuple(reversed(self.shape)), tuple(reversed(self.strides)))
    
    def sum(self, axis):
        if axis == None:
            return sum(self.data)
        
        # initialize empty array with the target shape
        out_shape = tuple(s for i, s in enumerate(self.shape) if i != axis)
        if len(out_shape) == 0:
            out_shape = (1,)
            
        result = NDArray.zeros(out_shape)
        # sum across the given axis
        ranges = [range(s) for s in self.shape]
        for idx in itertools.product(*ranges):
            out_idx = tuple(s for i, s in enumerate(idx) if i != axis)
            if not out_idx:
                out_idx = (0,)
            result[out_idx] += self[idx]

        return result
    
    def expand(self, new_shape):
        assert len(self.shape) == len(new_shape), "ndim mismatch"
        for i, (s, sn) in enumerate(zip(self.shape, new_shape)):
            assert s == sn or s == 1, f"incompatible shapes at dim {i}"

        result = NDArray.zeros(new_shape)

        ranges = [range(s) for s in new_shape]
        for idx in itertools.product(*ranges):
            src_idx = tuple(0 if self.shape[i] == 1 else idx[i] for i in range(len(idx)))
            result[idx] = self[src_idx]

        return result
    
    def __matmul__(self, other):
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(f"Expected Shape 2D, got {self.shape} and {other.shape}")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")
        
        m, n, p = self.shape[0], self.shape[1], other.shape[1]
        result = NDArray([[0] * p for _ in range(m)])

        for i in range(m):
            for j in range(p):
                result[(i, j)] = sum(self[(i, k)] * other[(k, j)] for k in range(n))
        return result
    
    def _broadcast_with(self, other):
        left = self
        right = other

        ll, lr = len(left.shape), len(right.shape)
        if ll > lr:
            right = right.reshape((1,) * (ll - lr) + right.shape)
        elif lr > ll:
            left = left.reshape((1,) * (lr - ll) + left.shape)

        result_shape = tuple(max(l, r) for l, r in zip(left.shape, right.shape))
        left = left.expand(result_shape)
        right = right.expand(result_shape)

        return left, right, result_shape
    
    def _binary_op(self, other, op):
        if not isinstance(other, NDArray):
            result = NDArray(self.data[:], self.shape)
            for i in range(len(result.data)):
                result.data[i] = op(self.data[i], other)
            return result
        
        left, right, result_shape = self._broadcast_with(other)
        result = NDArray.zeros(result_shape)

        for idx in itertools.product(*[range(s) for s in result_shape]):
            result[idx] = op(left[idx], right[idx])
        
        return result
    
    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)
    
    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)
    
    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)
    
    def __pow__(self, other):
        return self._binary_op(other, lambda a, b: a ** b)

    def __isub__(self, other):
        for i in range(len(self.data)):
            self.data[i] -= other.data[i]
        return self

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return self.data[self._flat_index(indices)]
    
    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.data[self._flat_index(indices)] = value
    
    def __repr__(self):
        return f"NDArray(Shape={self.shape}, Data={self.data})"
