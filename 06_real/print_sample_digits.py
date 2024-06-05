# Print a few lines of MNIST digits and their labels.

import mnist
import matplotlib.pyplot as plt
import random

X = mnist.load_images("../data/mnist/train-images-idx3-ubyte.gz")
Y = mnist.load_labels("../data/mnist/train-labels-idx1-ubyte.gz").flatten()

plt.ion()
plt.show()
_, ax = plt.subplots()

NUMBER_OF_EXAMPLES = 10
for i in range(NUMBER_OF_EXAMPLES):
    idx = random.randint(0, X.shape[0])
    ax.set_title("Label: %d" % Y[idx], fontsize="20")
    ax.imshow(X[idx].reshape((28, 28)), cmap="Greys")
    input("Press <Enter>...")
