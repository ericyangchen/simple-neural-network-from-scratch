import numpy as np

from .layer import FC, Conv2D
from .activation import Activation
from .optimizer import SGD, Momentum


class FullyConnectedNeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_type,
        learning_rate,
        optimizer,
    ):
        self.fc1 = FC(input_size, hidden_size, learning_rate, optimizer)
        self.activation1 = Activation(activation_type)

        self.fc2 = FC(hidden_size, hidden_size, learning_rate, optimizer)
        self.activation2 = Activation(activation_type)

        self.fc3 = FC(hidden_size, output_size, learning_rate, optimizer)
        self.activation3 = Activation(activation_type)

    def forward(self, input):
        x = self.fc1.forward(input)
        x = self.activation1.forward(x)

        x = self.fc2.forward(x)
        x = self.activation2.forward(x)

        x = self.fc3.forward(x)
        x = self.activation3.forward(x)

        return x

    def backward(self, grad):
        # grad = dE/dY

        grad = self.activation3.backward(grad)
        grad = self.fc3.backward(grad)

        grad = self.activation2.backward(grad)
        grad = self.fc2.backward(grad)

        grad = self.activation1.backward(grad)
        grad = self.fc1.backward(grad)


class ConvolutionalNeuralNetwork:
    def __init__(
        self,
        input_channel,
        hidden_channel,
        output_channel,
        kernel_size,
        stride,
        padding,
        output_size,
        activation_type,
        learning_rate,
        optimizer,
    ):
        self.conv1 = Conv2D(
            input_channel,
            hidden_channel,
            kernel_size,
            stride,
            padding,
            learning_rate,
            optimizer,
        )
        self.activation1 = Activation(activation_type)

        self.conv2 = Conv2D(
            hidden_channel,
            output_channel,
            kernel_size,
            stride,
            padding,
            learning_rate,
            optimizer,
        )
        self.activation2 = Activation(activation_type)

        self.fc1 = FC(output_size, output_size, learning_rate, optimizer)
        self.activation2 = Activation(activation_type)

    def forward(self, input):
        x = self.conv1.forward(input)
        x = self.activation1.forward(x)

        x = self.conv2.forward(x)
        x = self.activation2.forward(x)

        x = self.fc1.forward(x)
        x = self.activation2.forward(x)

        return x

    def backward(self, grad):
        # grad = dE/dY

        grad = self.activation2.backward(grad)
        grad = self.fc1.backward(grad)

        grad = self.activation2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.activation1.backward(grad)
        grad = self.conv1.backward(grad)