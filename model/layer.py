import numpy as np
from .optimizer import SGD, Momentum


class FC:
    def __init__(self, input_size, output_size, learning_rate, optimizer):
        self.learning_rate = learning_rate

        self.weight = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

        self.input = None

        if optimizer == "SGD":
            self.optimizer = SGD(learning_rate)
        elif optimizer == "Momentum":
            self.optimizer = Momentum(self.weight, self.bias, learning_rate, 0.9)

    def forward(self, input):
        self.input = input

        output = np.dot(self.input, self.weight) + self.bias

        return output

    # output_error: dE/dY
    def backward(self, output_error):
        # weight_error: dE/dW = dE/dY * dY/dW
        weight_error = np.dot(self.input.T, output_error)
        # bias_error: dE/dB = dE/dY * dY/dB
        bias_error = np.sum(output_error, axis=0)

        # input_error: dE/dX = dE/dY * dY/dX
        input_error = np.dot(output_error, self.weight.T)

        # Update weights and biases
        self.weight -= self.optimizer.step_weight(weight_error)
        self.bias -= self.optimizer.step_bias(bias_error)

        return input_error


# implement a convolutional layer with numpy
class Conv2D:
    pass
