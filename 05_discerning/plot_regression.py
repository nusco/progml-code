# Plot an example of linear regression.

import numpy as np


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b


X = np.array([13., 2., 14., 23., 13., 1., 18., 10., 26., 3., 3., 21., 7.,
              22., 2., 27., 6., 10., 18., 15., 9., 26., 8., 15., 10., 21.])
Y = np.array([12., 3., 11., 16., 8., 3., 12., 4., 13., 2., 3., 11., 6.,
              14., 1., 16., 3., 5., 13., 10., 8., 12., 7., 14., 8., 13.])
w, b = train(X, Y, iterations=100000, lr=0.001)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.plot(X, Y, "bo")
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
x_edge, y_edge = 30, 20
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()
