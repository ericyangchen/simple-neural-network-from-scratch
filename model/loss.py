import numpy as np

# E = 1/N * (y - y')^2
def MSE_loss(input, target):
    return np.mean(np.square(input - target))


# dE/dY = 2 * (y - y') / N
def MSE_loss_backward(input, target):
    return 2 * (input - target) / input.size
