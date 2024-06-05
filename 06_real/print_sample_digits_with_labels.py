# Print individual MNIST digits and their labels.

import mnist
import numpy as np
import matplotlib.pyplot as plt
import random

X = mnist.load_images("../data/mnist/train-images-idx3-ubyte.gz")
Y = mnist.load_labels("../data/mnist/train-labels-idx1-ubyte.gz")

ROWS = 3
COLUMNS = 24
fig = plt.figure()
for i in range(ROWS * COLUMNS):
    ax = fig.add_subplot(ROWS, COLUMNS, i + 1)
    ax.axis('off')
    idx = random.randint(0, X.shape[0])
    ax.set_title(Y[idx, 0], fontsize="13")
    ax.imshow(X[idx].reshape((28, 28)), cmap="Greys")
plt.show()
