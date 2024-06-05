# Plot the path of GD on a function of one input, to show how a bigger learning
# rate makes the system learn faster, but a learning rate that's too large
# makes the GD algorithm diverge.

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Change this value to visualize GD
# with different values of lr:
lr = 0.0001
# ------------------------------------


def predict(X, w):
    return X * w


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w) - Y))


X = np.array([13, 2, 14, 23, 13, 1, 18, 10, 26, 3, 3,
              21, 7, 22, 2, 27, 6, 10, 18, 15, 9, 26,
              8, 15, 10, 21, 5, 6, 13, 13])
Y = np.array([33, 16, 32, 51, 27, 16, 34, 17, 29, 15,
              15, 32, 22, 37, 13, 44, 16, 21, 37, 30,
              26, 34, 23, 39, 27, 37, 17, 18, 25, 23])

# Compute losses for w ranging from -1 to 4
weights = np.linspace(-1.0, 4.0, 200)
losses = [loss(X, Y, w) for w in weights]

# Plot weights and losses
plt.axis([-1, 4, 0, 1000])
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
plt.plot(weights, losses, color="black")

# Put a green cross on the minimum loss
min_index = np.argmin(losses)
plt.plot(weights[min_index], losses[min_index], "gX", markersize=13)

NUMBER_OF_STEPS = 5
w = 0.5
plt.plot(w, loss(X, Y, w), "rX", markersize=13)
for i in range(NUMBER_OF_STEPS):
    earlier_w = w
    w -= gradient(X, Y, w) * lr
    plt.plot(w, loss(X, Y, w), "ro", markersize=9)
    plt.plot([earlier_w, w], [loss(X, Y, earlier_w),
             loss(X, Y, w)], color="red")
plt.xlabel("lr = %f" % lr, fontsize=30)
plt.show()
