class NDArray:
    def __init__(self, data):
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
    
    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        return self.data[flat_index]
    
    def __repr__(self):
        return f"NDArray(Shape={self.shape}, Data={self.data})"