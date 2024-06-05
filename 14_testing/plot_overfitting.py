# Plot the losses on the training and test set during GD, to show the
# effect of overfitting.

import numpy as np
import mnist_two_sets as data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)


def sigmoid_gradient(sigmoid):
    return np.multiply(sigmoid, (1 - sigmoid))


def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return (y_hat, h)


def back(X, Y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - Y)) / X.shape[0]
    w1_gradient = np.matmul(prepend_bias(X).T,
                            np.matmul(y_hat - Y, w2[1:].T) *
                            sigmoid_gradient(h)) / X.shape[0]
    return (w1_gradient, w2_gradient)


def classify(X, w1, w2):
    y_hat, _ = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def accuracy(X, Y_unencoded, w1, w2):
    return np.average(classify(X, w1, w2) == Y_unencoded) * 100.0


def initialize_weights(n_input_variables, n_hidden_nodes, n_classes):
    np.random.seed(1234)

    w1_rows = n_input_variables + 1
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)

    return (w1, w2)


# This loss() takes different parameters than the ones in other source files
def loss(X, Y, w1, w2):
    y_hat, _ = forward(X, w1, w2)
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]


# This train() stores the loss on both the training and the test sets at each
# step. A the end, it returns those histories to the caller.
# Different from train() functions in other source files, it expects that
# _both_ Y_train and Y_test are one hot encoded.
def train(X_train, Y_train, X_test, Y_test, n_hidden_nodes, iterations, lr):
    n_input_variables = X_train.shape[1]
    n_classes = Y_train.shape[1]
    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    training_losses = []
    test_losses = []
    for i in range(iterations):
        y_hat_train, h = forward(X_train, w1, w2)
        y_hat_test, _ = forward(X_test, w1, w2)
        w1_gradient, w2_gradient = back(X_train, Y_train, y_hat_train, w2, h)
        w1 = w1 - (w1_gradient * lr)
        w2 = w2 - (w2_gradient * lr)

        training_loss = -np.sum(Y_train * np.log(y_hat_train)) / \
            Y_train.shape[0]
        training_losses.append(training_loss)
        test_loss = -np.sum(Y_test * np.log(y_hat_test)) / Y_test.shape[0]
        test_losses.append(test_loss)
        print("%5d > Training loss: %.5f - Test loss: %.5f" %
              (i, training_loss, test_loss))
    return (training_losses, test_losses, w1, w2)


training_losses, test_losses, w1, w2 = train(data.X_train,
                                             data.Y_train,
                                             data.X_test,
                                             data.one_hot_encode(data.Y_test),
                                             n_hidden_nodes=200,
                                             iterations=10000,
                                             lr=0.01)

training_accuracy = accuracy(data.X_train, data.Y_train_unencoded, w1, w2)
test_accuracy = accuracy(data.X_test, data.Y_test, w1, w2)
print("Training accuracy: %.2f%%, Test accuracy: %.2f%%" %
      (training_accuracy, test_accuracy))

plt.plot(training_losses, label='Training set', color='blue', linestyle='-')
plt.plot(test_losses, label='Test set', color='green', linestyle='--')
plt.xlabel("Iterations", fontsize=30)
plt.ylabel("Loss", fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=30)
plt.show()
