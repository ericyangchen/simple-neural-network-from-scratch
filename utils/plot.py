import numpy as np
import matplotlib.pyplot as plt

def show_result(x, y, pred_y, plot_name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].set_title("Ground truth", fontsize=10)
    for i in range(x.shape[0]):
        if y[i] == 0:
            axs[0].plot(x[i][0], x[i][1], "ro")
        else:
            axs[0].plot(x[i][0], x[i][1], "bo")

    axs[1].set_title("Prediction result", fontsize=10)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            axs[1].plot(x[i][0], x[i][1], "ro")
        else:
            axs[1].plot(x[i][0], x[i][1], "bo")

    if plot_name:
        plt.savefig(f"assets/{plot_name}")

    plt.close()


def show_learning_curve_and_accuracy_curve(loss_history, acc_history, plot_name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting loss
    axs[0].plot(np.arange(len(loss_history)), loss_history, "b")
    axs[0].set_title("Learning Curve", fontsize=10)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    # Plotting accuracy
    axs[1].plot(np.arange(len(acc_history)), acc_history, "r")
    axs[1].set_title("Accuracy Curve", fontsize=10)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    if plot_name:
        plt.savefig(f"assets/{plot_name}")

    plt.close()
