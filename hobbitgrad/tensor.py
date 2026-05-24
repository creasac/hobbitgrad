from hobbitgrad.array import NDArray

class Tensor:
    def __init__(self, data):
        if not isinstance(data, NDArray):
            self.data = NDArray([data] if not isinstance(data, list) else data)
        else:
            self.data = data
        self.grad = NDArray.zeros(self.data.shape)
        self._prev = set()
        self._backward = lambda: None

    def backward(self):
        topo = []
        visited = set()

        def build(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build(parent)
                topo.append(tensor)
        
        build(self)
        self.grad = NDArray.ones(self.data.shape)
        for tensor in reversed(topo):
            tensor._backward()

    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad @ other.data.__transpose__()
            other.grad += self.data.__transpose__() @ out.grad

        out._backward = _backward
        return out
