# Plot the loss during a run of stochastic GD.

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


def train(X, Y, iterations, lr, precision, initial_w, initial_b):
    w, b = initial_w, initial_b

    previous_loss = loss(X, Y, w, b)
    history = [[w, b, previous_loss]]
    for i in range(0, iterations):
        for j in range(0, X.size):
            w_gradient, b_gradient = loss_gradients(X[j], Y[j], w, b)
            w -= lr * w_gradient
            b -= lr * b_gradient

            current_loss = loss(X[j], Y[j], w, b)
            history.append([w, b, previous_loss])

            if (abs(current_loss - previous_loss) < precision):
                return w, b, history

            previous_loss = current_loss

    return w, b, history


# Set up data, train model
X = np.array([13.0, 12.0, 10.0, 1.0, 3.0, 1.0, 18.0, 10.0, 31.0, 3.0, 21.0])
Y = np.array([33.0, 16.0, 32.0, 51.0, 27.0,
              16.0, 34.0, 17.0, 15.0, 15.0, 32.0])
w, b, history = train(X, Y, iterations=3, lr=0.0001,
                      precision=0.0001, initial_w=-10, initial_b=-100)

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
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
ax.plot_surface(W, B, L, cmap=cm.gnuplot, linewidth=0, antialiased=True)

# Mark endpoint
plt.plot([history_w[-1]], [history_b[-1]], [history_loss[-1]],
         "gX", markersize=16)

# Mark startpoint and path
plt.plot([history_w[0]], [history_b[0]], [history_loss[0]], "wo")
plt.plot(history_w, history_b, history_loss, color="w",
         linestyle="dashed", linewidth=1)

plt.show()
