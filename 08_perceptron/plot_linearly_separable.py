# Print a few examples of linearly separable and non-separable datasets.
#
# This is the only code in Part I of the book that uses the scikit-learn
# library. I didn't ask you to install scikit-learn in the first chapter,
# because this program is only used to generate the diagrams in the
# "Perceptron" chapter. If you want to run this code yourself, then
# install scikit-learn first, either with conda:
#
#   conda install scikit-learn=0.24.1
#
# ...or with pip:
#
#   pip3 install scikit-learn==0.24.1

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


def plot(points, clusters):
    blue_points = points[clusters == 0]
    plt.plot(blue_points[:, 0], blue_points[:, 1], 'bs')

    green_points = points[clusters == 1]
    plt.plot(green_points[:, 0], green_points[:, 1], 'g^')

    red_points = points[clusters == 2]
    plt.plot(red_points[:, 0], red_points[:, 1], 'ro')

    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])

    plt.show()
    input("Press <Enter>...")
    plt.close()


plt.ion()

# linearly separable with two classes
plt.axis([-15, 5, -12, 12])
points, clusters = make_blobs(n_samples=100,
                              centers=2,
                              n_features=2,
                              cluster_std=2.5,
                              random_state=1)
plot(points, clusters)

# linearly separable with three classes
points, clusters = make_blobs(n_samples=150,
                              centers=[[1, 1], [-1, -1], [1, -2]],
                              n_features=2,
                              cluster_std=0.3,
                              random_state=123)
plot(points, clusters)

# non-linearly separable
points, clusters = make_blobs(n_samples=100,
                              centers=2,
                              n_features=2,
                              cluster_std=3.8,
                              random_state=1)
plot(points, clusters)

# very non-linearly separable
points, clusters = make_moons(n_samples=100,
                              noise=0.1,
                              random_state=1)
plot(points, clusters)
