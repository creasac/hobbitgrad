# hobbitgrad

hobbitgrad is a tiny educational autograd library written in pure Python. It includes a small `NDArray`, a `Tensor` type with backpropagation, a linear layer, SGD, and mean squared error loss.

## Install

```bash
pip install hobbitgrad
```

## Example

```python
from hobbitgrad import Linear, SGD, Tensor, mse

x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [0], [0], [1]])

model = Linear(2, 1)
optimizer = SGD(model.parameters(), lr=0.1)

for _ in range(100):
    pred = model.forward(x)
    loss = mse(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(loss.data.data[0])
```

## Current Scope

hobbitgrad currently supports a small set of tensor operations, broadcasting, matrix multiplication, scalar reductions, a linear layer, SGD, and MSE loss.
