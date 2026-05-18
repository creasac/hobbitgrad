class Tensor:
    def __init__(self, data):
        self.data = data # NDArray
        self.grad = 0
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
        self.grad = 1
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
