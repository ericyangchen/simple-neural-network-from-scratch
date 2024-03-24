import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step_weight(self, weight_error):
        return self.learning_rate * weight_error

    def step_bias(self, bias_error):
        return self.learning_rate * bias_error


class Momentum:
    def __init__(self, weight, bias, learning_rate, momentum):
        self.learning_rate = learning_rate

        self.momentum = momentum

        self.velocity_weight = np.zeros_like(weight)
        self.velocity_bias = np.zeros_like(bias)

    def step_weight(self, weight_error):
        self.velocity_weight = (
            self.momentum * self.velocity_weight - self.learning_rate * weight_error
        )

        return -self.velocity_weight

    def step_bias(self, bias_error):
        self.velocity_bias = (
            self.momentum * self.velocity_bias - self.learning_rate * bias_error
        )

        return -self.velocity_bias
