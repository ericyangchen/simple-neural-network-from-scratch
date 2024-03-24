import argparse
import numpy as np

from dataset.generator import generate_data
from model.model import FullyConnectedNeuralNetwork, ConvolutionalNeuralNetwork
from model.loss import MSE_loss, MSE_loss_backward
from utils.random import set_random_seed
from utils.metrics import accuracy
from utils.plot import show_result, show_learning_curve_and_accuracy_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple neural network model.")
    parser.add_argument(
        "--dataset", type=str, default="linear", help="Dataset to use. [linear | xor]"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FullyConnectedNeuralNetwork",
        help="Model to train. [FullyConnectedNeuralNetwork | ConvolutionalNeuralNetwork]",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        help="Activation function for the model. [sigmoid | relu | tanh | none]",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100000,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate for the model."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=16, help="Hidden size for the model."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer for the model. [SGD | Momentum]",
    )

    args = parser.parse_args()

    return args


def train(x, y, model, epochs, learning_rate):
    loss_history = []
    accuracy_history = []
    for epoch in range(1, epochs + 1):
        y_pred = model.forward(x)

        loss = MSE_loss(y_pred, y)
        loss_grad = MSE_loss_backward(y_pred, y)
        model.backward(loss_grad)

        y_pred_binary = y_pred > 0.5
        acc = accuracy(y_pred_binary, y)

        loss_history.append(loss)
        accuracy_history.append(acc)

        if epoch % 5000 == 0:
            print(f"Epoch {epoch:7}, Loss {loss:.16f}, Accuracy {acc:.2f}")

    return loss_history, accuracy_history, y_pred, y_pred_binary, model


def test(x, y, model):
    y_pred = model.forward(x)
    y_pred_binary = y_pred > 0.5

    for i in range(x.shape[0]):
        print(
            f"iter {i+1:3} | Ground truth: {y[i][0]} | Prediction: {y_pred[i][0]:.16f} |"
        )

    print("\nTesting accuracy: ", accuracy(y_pred_binary, y))


if __name__ == "__main__":
    args = parse_args()

    # set random
    set_random_seed()

    # dataset
    x, y = generate_data(args.dataset)

    # model
    model = None
    if args.model == "FullyConnectedNeuralNetwork":
        model = FullyConnectedNeuralNetwork(
            input_size=x.shape[1],
            hidden_size=args.hidden_size,
            output_size=1,
            activation_type=args.activation,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
        )
    elif args.model == "ConvolutionalNeuralNetwork":
        model = ConvolutionalNeuralNetwork()
    else:
        raise ValueError("Invalid model type.")

    # training
    loss_history, accuracy_history, y_pred, y_pred_binary, model = train(
        x, y, model, args.epochs, args.learning_rate
    )

    # testing (with the same data)
    test(x, y, model)

    # visualization
    show_learning_curve_and_accuracy_curve(
        loss_history,
        accuracy_history,
        f"momentum_optimizer/learning_curve_and_accuracy_curve_{args.dataset}_{args.model}_{args.activation}_{args.epochs}_{args.learning_rate}_{args.hidden_size}.png",
    )
    show_result(
        x,
        y,
        y_pred_binary,
        f"momentum_optimizer/result_{args.dataset}_{args.model}_{args.activation}_{args.epochs}_{args.learning_rate}_{args.hidden_size}.png",
    )
