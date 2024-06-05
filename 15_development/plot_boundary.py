# Plot the decision boundary a neural network with different numbers of hidden
# nodes, to show how the boundary becomes more complicated as the number of
# hidden nodes grow.

import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------------------------
# Change this value to experiment with
# different numbers of hidden nodes:
HIDDEN_NODES = 1
# ------------------------------------


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    result = np.zeros((n_labels, 2))
    for i in range(n_labels):
        result[i][Y[i]] = 1
    return result


x1, x2, y = np.loadtxt('non_linearly_separable.txt', skiprows=1, unpack=True)
X_train = X_test = np.column_stack((x1, x2))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encode(Y_train_unencoded)
w1, w2 = nn.train(X_train, Y_train,
                  X_test, Y_test,
                  n_hidden_nodes=HIDDEN_NODES,
                  epochs=100000,
                  batch_size=X_train.shape[0],
                  lr=0.1)


# Generate a mesh over one-dimensional data
# (The mesh() and plot_boundary() functionality were inspired by the
# documentation of the BSD-licensed scikit-learn library.)
def mesh(values):
    range = values.max() - values.min()
    padding_percent = 5
    padding = range * padding_percent * 0.01
    resolution = 1000
    interval = (range + 2 * range * padding) / resolution
    return np.arange(values.min() - padding, values.max() + padding, interval)


def plot_boundary(points, w1, w2):
    print("Calculating boundary...")
    # Generate a grid of points over the data
    x_mesh = mesh(points[:, 0])
    y_mesh = mesh(points[:, 1])
    grid_x, grid_y = np.meshgrid(x_mesh, y_mesh)
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    # Classify points in the grid
    classifications = nn.classify(grid, w1, w2).reshape(grid_x.shape)
    # Trace the decision boundary
    BLUE_AND_GREEN = ListedColormap(['#BBBBFF', '#BBFFBB'])
    plt.contourf(grid_x, grid_y, classifications, cmap=BLUE_AND_GREEN)


def plot_data_by_label(input_variables, labels, label_selector, symbol):
    points = input_variables[(labels == label_selector).flatten()]
    plt.plot(points[:, 0], points[:, 1], symbol, markersize=4)


plot_boundary(X_train, w1, w2)
plot_data_by_label(X_train, Y_train_unencoded, 0, 'bo')
plot_data_by_label(X_train, Y_train_unencoded, 1, 'go')
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
plt.xlabel("%d hidden nodes" % HIDDEN_NODES, fontsize=20)
plt.show()
