from hobbitgrad.tensor import Tensor
from hobbitgrad.array import NDArray

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(NDArray.randn((in_features, out_features)))
        self.b = Tensor(NDArray.zeros((out_features,)))

    def forward(self, x):
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]
