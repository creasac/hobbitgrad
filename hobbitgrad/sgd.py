from hobbitgrad.array import NDArray

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params:
            p.grad = NDArray.zeros(p.data.shape)