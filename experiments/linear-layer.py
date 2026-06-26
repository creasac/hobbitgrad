from hobbitgrad.array import NDArray
from hobbitgrad.tensor import Tensor
from hobbitgrad.linear import Linear
from hobbitgrad.sgd import SGD
from hobbitgrad.loss import mse
import random

# exp 1
# x = Tensor(NDArray.randn((4, 3)))
# layer = Linear(3, 5)
# out = layer.forward(x)
# print("out.data.shape", out.data.shape)

# loss = out.sum()
# loss.backward()
# print("loss.data.shape", loss.data.shape)

# print("layer.W.data.shape", layer.W.data.shape)
# print("layer.b.data.shape", layer.b.data.shape)
# print("x.grad.shape", x.grad.shape)


# exp 2
# x = Tensor(NDArray([[0,0], [0,1], [1,0], [1,1]]))
# yxor = Tensor(NDArray([[0], [1], [1], [0]]))

# model = Linear(2, 1)
# optimizer = SGD(model.parameters(), lr=0.01)

# for epoch in range(100):
#     pred = model.forward(x)
#     loss = mse(pred, yxor)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if epoch % 10 == 0:
#         print(f"epoch {epoch}, loss {loss.data.data[0]:.4f}")

# [weights, biases] = model.parameters()
# print("weights:", weights.data)
# print("biases:", biases.data)


# exp3
n = 1000
total_correct = 0
for i in range(n):
    r1, r2 = random.randint(0,255), random.randint(0,255)
    x = Tensor(NDArray([[r1]]))
    y = Tensor(NDArray([[r2]]))

    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.000001)

    losse = 100
    epoch = 0
    while losse > 0.01:
        pred = model.forward(x)
        loss = mse(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losse = loss.data.data[0]
        epoch += 1
        # if epoch % 1000 == 0:
        #     print(f"epoch {epoch}, loss {loss.data.data[0]:.4f}")

    # [weights, biases] = model.parameters()
    # print("weights:", weights.data)
    # print("biases:", biases.data)

    finaly = round(model.forward(x).data.data[0])
    if r2 == finaly:
        total_correct += 1
        print("correct at r1, r2, y = ", r1, r2, finaly)
    else:
        print("incorrect at r1, r2, y = ", r1, r2, finaly)

print("total correct = ", total_correct)