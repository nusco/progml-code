# Plot the model function of a perceptron on a dataset with two input variables.


import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()

np.random.seed(123)  # Make this code deterministic

def one_hot_encode(Y):
    n_labels = Y.shape[0]
    result = np.zeros((n_labels, 2))
    for i in range(n_labels):
        result[i][Y[i]] = 1
    return result


# Uncomment one of the next three lines to decide which dataset to load
# x1, x2, y = np.loadtxt('linearly_separable.txt', skiprows=1, unpack=True)
x1, x2, y = np.loadtxt('non_linearly_separable.txt', skiprows=1, unpack=True)
# x1, x2, y = np.loadtxt('circles.txt', skiprows=1, unpack=True)

# Train classifier
X_train = X_test = np.column_stack((x1, x2))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encode(Y_train_unencoded)
w1, w2 = nn.train(X_train, Y_train,
                  X_test, Y_test,
                  n_hidden_nodes=10, iterations=100000, lr=0.3)

# Plot the axes
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().gca(projection="3d")
ax.set_zticks([0, 0.5, 1])
ax.set_xlabel("Input A", labelpad=15, fontsize=30)
ax.set_ylabel("Input B", labelpad=15, fontsize=30)
ax.set_zlabel("ŷ", labelpad=5, fontsize=30)

# Plot the data points
blue_squares = X_train[(Y_train_unencoded == 0).flatten()]
ax.scatter(blue_squares[:, 0], blue_squares[:, 1], 0, c='b', marker='s')
green_triangles = X_train[(Y_train_unencoded == 1).flatten()]
ax.scatter(green_triangles[:, 0], green_triangles[:, 1], 1, c='g', marker='^')

# Plot the model
MARGIN = 0.5
MESH_SIZE = 1000  # This model has a lot of detail, so we need a hi-res mesh
x, y = np.meshgrid(np.linspace(x1.min() - MARGIN, x1.max() + MARGIN, MESH_SIZE),
                   np.linspace(x2.min() - MARGIN, x2.max() + MARGIN, MESH_SIZE))
grid = zip(np.ravel(x), np.ravel(y))
# Calculate all the outputs of forward(), in the format (y_hat, h):
forwards = [nn.forward(np.column_stack(([i], [j])), w1, w2) for i, j in grid]
# For each (y_hat, y), keep only the second column of y_hat:
z = np.array([y_hat for y_hat, h in forwards])[:, 0, 1]
z = z.reshape((MESH_SIZE, MESH_SIZE))
ax.plot_surface(x, y, z, alpha=0.75, cmap=cm.winter,
                linewidth=0, antialiased=True)
plt.show()
