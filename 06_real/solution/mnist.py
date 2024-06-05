import numpy as np
import gzip
import struct


def load_images(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    # Insert a column of 1s in the position 0 of X.
    # (“axis=1” stands for: “insert a column, not a row”)
    return np.insert(X, 0, 1, axis=1)


# 60000 images, each 785 elements (1 bias + 28 * 28 pixels)
X_train = prepend_bias(load_images("../../data/mnist/train-images-idx3-ubyte.gz"))

# 10000 images, each 785 elements, with the same structure as X_train
X_test = prepend_bias(load_images("../../data/mnist/t10k-images-idx3-ubyte.gz"))


def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)

def encode_digit(Y, digit):
    encoded_Y = np.zeros_like(Y)
    n_labels = Y.shape[0]
    for i in range(n_labels):
        if Y[i] == digit:
            encoded_Y[i][0] = 1
    return encoded_Y

TRAINING_LABELS = load_labels("../../data/mnist/train-labels-idx1-ubyte.gz")
TEST_LABELS = load_labels("../../data/mnist/t10k-labels-idx1-ubyte.gz")

# Load a list of 10 training sets and a matching list of 10 testing sets, each
# encoding one digit.

Y_train = []
Y_test = []

for digit in range(10):
    Y_train.append(encode_digit(TRAINING_LABELS, digit))
    Y_test.append(encode_digit(TEST_LABELS, digit))
