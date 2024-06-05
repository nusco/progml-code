# Load the Echidna dataset.

import os
import numpy as np


def load(filename):
    data = np.loadtxt(filename, skiprows=1, unpack=True).T
    np.random.seed(12345)
    np.random.shuffle(data)
    x_raw = data[:, 0:2]
    x_min = x_raw.min(axis=0)
    x_max = x_raw.max(axis=0)
    # Rescale data between -0.5 and 0.5
    x = (x_raw - x_min) / (x_max - x_min) - 0.5
    y = data[:, 2].astype(int).reshape(-1, 1)
    return (x, y)


current_directory = os.path.dirname(__file__)
filename = os.path.join(current_directory, './echidna.txt')
X, Y = load(filename)
X_train, X_validation, X_test = np.split(X, 3)
Y_train, Y_validation, Y_test = np.split(Y, 3)
