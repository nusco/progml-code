# Plot an example of classification.

import numpy as np
import matplotlib.pyplot as plt
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


x1, _, _, y = np.loadtxt("police.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=1000000, lr=0.01)

plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Police Call", fontsize=30)
MARGIN = 3
left_edge = X[:, 1].min() - MARGIN
right_edge = X[:, 1].max() + MARGIN
inputs = np.linspace(left_edge - MARGIN, right_edge + MARGIN, 2500)
x_values = np.column_stack((np.ones(inputs.size), inputs.reshape(-1, 1)))

# Uncomment one of the two lines below to plot the model with/without the
# rounding introduced by classify()
y_values = forward(x_values, w)   # no rounding
# y_values = classify(x_values, w)  # rounded

plt.axis([left_edge - MARGIN, right_edge + MARGIN, -0.05, 1.05])
plt.plot(x_values[:, 1], y_values, color="g")
plt.show()
