from hobbitgrad.array import NDArray
from hobbitgrad.tensor import Tensor
from hobbitgrad.linear import Linear

x = Tensor(NDArray.randn((4, 3)))
layer = Linear(3, 5)
out = layer.forward(x)
print("out.data.shape", out.data.shape)

loss = out.sum()
loss.backward()
print("loss.data.shape", loss.data.shape)

print("layer.W.data.shape", layer.W.data.shape)
print("layer.b.data.shape", layer.b.data.shape)
print("x.grad.shape", x.grad.shape)