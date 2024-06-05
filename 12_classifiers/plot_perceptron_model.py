# Plot the model function of a perceptron on a dataset with two input variables.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


def classify(X, w):
    return np.round(forward(X, w))


def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        w -= gradient(X, Y, w) * lr
    return w


# Uncomment one of the next three lines to decide which dataset to load
x1, x2, y = np.loadtxt("linearly_separable.txt", skiprows=1, unpack=True)
# x1, x2, y = np.loadtxt('non_linearly_separable.txt', skiprows=1, unpack=True)
# x1, x2, y = np.loadtxt('circles.txt', skiprows=1, unpack=True)

# Train classifier
X = np.column_stack((np.ones(x1.size), x1, x2))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)

# Plot the axes
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().gca(projection="3d")
ax.set_zticks([0, 0.5, 1])
ax.set_xlabel("Input A", labelpad=15, fontsize=30)
ax.set_ylabel("Input B", labelpad=15, fontsize=30)
ax.set_zlabel("ŷ", labelpad=5, fontsize=30)

# Plot the data points
blue_squares = X[(Y == 0).flatten()]
ax.scatter(blue_squares[:, 1], blue_squares[:, 2], 0, c='b', marker='s')
green_triangles = X[(Y == 1).flatten()]
ax.scatter(green_triangles[:, 1], green_triangles[:, 2], 1, c='g', marker='^')

# Plot the model
MARGIN = 0.5
MESH_SIZE = 50
x, y = np.meshgrid(np.linspace(x1.min() - MARGIN, x1.max() + MARGIN, MESH_SIZE),
                   np.linspace(x2.min() - MARGIN, x2.max() + MARGIN, MESH_SIZE))
grid = zip(np.ravel(x), np.ravel(y))
z = np.array([forward(np.column_stack(([1], [i], [j])), w) for i, j in grid])
z = z.reshape((MESH_SIZE, MESH_SIZE))
ax.plot_surface(x, y, z, alpha=0.75, cmap=cm.winter,
                linewidth=0, antialiased=True)
plt.show()
