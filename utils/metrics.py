import numpy as np

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)
