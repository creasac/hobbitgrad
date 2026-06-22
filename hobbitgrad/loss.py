from hobbitgrad.tensor import Tensor

def mse(predictions, target):
    n = predictions.data._total_elements(predictions.data.shape)
    return ((predictions - target) ** 2).sum() / n