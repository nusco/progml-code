# Plot the Echidna dataset---all of it, and then split into training,
# validation and test sets.

import numpy as np
import matplotlib.pyplot as plt
import echidna as data


def plot_data_by_label(input_variables, labels, label_selector, symbol):
    points = input_variables[(labels == label_selector).flatten()]
    plt.plot(points[:, 0], points[:, 1], symbol, markersize=4)


def plot_data(title, x, y):
    RANGE = 0.55
    plt.xlim(-RANGE, RANGE)
    plt.ylim(-RANGE, RANGE)
    plot_data_by_label(x, y, 0, 'bs')
    plot_data_by_label(x, y, 1, 'g^')
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.title(title, fontsize=15)
    plt.ion()
    plt.show()
    input("Enter to continue...")
    plt.close()


plot_data("All data", data.X, data.Y)
plot_data("Training set", data.X_train, data.Y_train)
plot_data("Validation set", data.X_validation, data.Y_validation)
plot_data("Test set", data.X_test, data.Y_test)
