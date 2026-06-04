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
            def reshape_to_sum(grad, target_shape):
                # pad target shape with leading 1s to match grad ndim
                padded = (1,) * (len(grad.shape) - len(target_shape)) + target_shape
                for i in range(len(grad.shape) - 1, -1, -1):
                    if padded[i] == 1 and grad.shape[i] != 1:
                        grad = grad.sum(i)
                return grad.reshape(target_shape)
            
            self.grad += reshape_to_sum(out.grad, self.data.shape)
            other.grad += reshape_to_sum(out.grad, other.data.shape)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad @ other.data.transpose()
            other.grad += self.data.transpose() @ out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(NDArray([self.data.sum(axis=None)]))
        out._prev = {self}

        def _backward():
            self.grad += NDArray.ones(self.data.shape) * out.grad.data[0]

        out._backward = _backward
        return out
