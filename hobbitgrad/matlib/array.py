class NDArray:
    def __init__(self, data, shape=None, strides=None):
        if shape is not None:
            self.data = data
            self.shape = shape
            self.strides = strides if strides is not None else self._compute_strides(shape)
        else:
            self.data, self.shape = self._flatten(data)
            self.strides = self._compute_strides(self.shape)

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
        import itertools
        ranges = [range(s) for s in self.shape]
        return [self.data[self._flat_index(idx)] for idx in itertools.product(*ranges)]
    
    def __reshape__(self, new_shape):
        assert self._total_elements(new_shape) == len(self.data), "Shape mismatch"
        data = self._contiguous_data() if not self._is_contiguous() else self.data
        return NDArray(data, new_shape)
    
    def __transpose__(self):
        return NDArray(self.data, tuple(reversed(self.shape)), tuple(reversed(self.strides)))
    
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
