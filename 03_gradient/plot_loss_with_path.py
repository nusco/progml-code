# Plot the loss on a dataset with two input variables, and the path of
# gradient descent across it.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def loss_gradients(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)


def train_with_history(X, Y, iterations, lr, precision, initial_w, initial_b):
    w, b = initial_w, initial_b

    previous_loss = loss(X, Y, w, b)
    history = [[w, b, previous_loss]]
    for i in range(0, iterations):
        w_gradient, b_gradient = loss_gradients(X, Y, w, b)
        w -= lr * w_gradient
        b -= lr * b_gradient

        current_loss = loss(X, Y, w, b)
        history.append([w, b, previous_loss])

        if (abs(current_loss - previous_loss) < precision):
            return w, b, history

        previous_loss = current_loss

    raise Exception("Couldn't converge within %d iterations" % iterations)


# Load data, train model
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b, history = train_with_history(X, Y, iterations=100000,
                                   lr=0.001, precision=0.000001,
                                   initial_w=-10, initial_b=-100)

# Prepare history
history = np.array(history)
history_w = history[:, 0]
history_b = history[:, 1]
history_loss = history[:, 2]

# Prepare matrices for 3D plot (W, B and L for weights, biases and losses)
MESH_SIZE = 20
weights = np.linspace(np.min(history_w) - 10, np.max(history_w) + 10,
                      MESH_SIZE)
biases = np.linspace(np.min(history_b) - 100, np.max(history_b) + 100,
                     MESH_SIZE)
W, B = np.meshgrid(weights, biases)
losses = np.array([loss(X, Y, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
L = losses.reshape((MESH_SIZE, MESH_SIZE))

# Plot surface
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
ax = plt.figure().gca(projection="3d")
ax.set_zticklabels(())
ax.set_xlabel("Weight", labelpad=20, fontsize=30)
ax.set_ylabel("Bias", labelpad=20, fontsize=30)
ax.set_zlabel("Loss", labelpad=5, fontsize=30)
ax.plot_surface(W, B, L, cmap=cm.gnuplot,
                linewidth=0, antialiased=True, color='black')

# Mark endpoint
plt.plot([history_w[-1]], [history_b[-1]], [history_loss[-1]],
         "gX", markersize=16)

# Display plot in interactive mode
plt.ion()
plt.show()
input("Enter to continue...")

# Mark startpoint and path
plt.plot([history_w[0]], [history_b[0]], [history_loss[0]], "wo")
plt.plot(history_w, history_b, history_loss, color="w", linestyle="dashed")

input("Enter to close...")
