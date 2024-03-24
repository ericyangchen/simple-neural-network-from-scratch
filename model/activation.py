import numpy as np

class Activation:
    def __init__(self, activation_type):
        self.type = activation_type
        self.activation_output = None

    def forward(self, x):
        if self.type == "sigmoid":
            self.activation_output = 1 / (1 + np.exp(-x))
        elif self.type == "relu":
            self.activation_output = np.maximum(0, x)
        elif self.type == "tanh":
            self.activation_output = np.tanh(x)
        elif self.type == "none":
            self.activation_output = x
        else:
            raise ValueError("Invalid activation function.")

        return self.activation_output

    def derivative_of_activation(self):
        y = self.activation_output

        if self.type == "sigmoid":
            return y * (1 - y)
        elif self.type == "relu":
            return 1.0 * (y > 0)
        elif self.type == "tanh":
            return 1 - y**2
        elif self.type == "none":
            return 1
        else:
            raise ValueError("Invalid activation function.")

    def backward(self, grad):
        return grad * self.derivative_of_activation()
