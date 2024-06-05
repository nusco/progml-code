# Utility functions to compare different configurations of the network.
# For a concrete example of how to use this, look at compare_batch_sizes.py.

import neural_network as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns


# This train() replaces the one in neural_network.py, and it's different from
# the original one in a few ways:
# * it goes on until a specified time has passed, rather than after a specified
#   number of epochs;
# * it does its job quietly instead of reporting the loss and accuracy at each
#   step;
# * it stores the loss and the time passed after each step, so that it can
#   return that history to the caller;
# * it also returns the number of training epochs and the total number of
#   gradient descent steps.
def train(X_train, Y_train, X_test, Y_test,
          n_hidden_nodes, lr, batch_size, time_in_seconds):
    n_input_variables = X_train.shape[1]
    n_classes = Y_train.shape[1]

    w1, w2 = nn.initialize_weights(n_input_variables,
                                   n_hidden_nodes, n_classes)
    x_batches, y_batches = nn.prepare_batches(X_train, Y_train, batch_size)

    start_time = time.time()
    times = []
    losses = []
    epochs = 0
    steps = 0
    while True:
        batch = 0
        while (batch < len(x_batches)):
            training_classifications, _ = nn.forward(X_train, w1, w2)
            training_loss = nn.loss(Y_train, training_classifications)
            times.append(np.floor(time.time() - start_time))
            losses.append(training_loss)

            time_passed = time.time() - start_time
            if time_passed > time_in_seconds:
                return (times, losses, epochs, steps)

            y_hat, h = nn.forward(x_batches[batch], w1, w2)
            w1_gradient, w2_gradient = nn.back(x_batches[batch],
                                               y_batches[batch],
                                               y_hat, w2, h)
            w1 = w1 - (w1_gradient * lr)
            w2 = w2 - (w2_gradient * lr)

            batch += 1
            steps += 1
        epochs += 1


def plot_loss(data, n_hidden_nodes, batch_size, lr,
              time_in_seconds, label, color, linestyle):
    print("Training:", label)
    times, losses, epochs, steps = train(data.X_train, data.Y_train,
                                         data.X_test, data.Y_test,
                                         n_hidden_nodes=n_hidden_nodes,
                                         batch_size=batch_size, lr=lr,
                                         time_in_seconds=time_in_seconds)
    print("  Loss: %.8f (%d epochs completed, %d total steps)" %
          (losses[-1], epochs, steps))
    plt.plot(times, losses, label=label, color=color, linestyle=linestyle)


def show_plot():
    sns.set()

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Seconds", fontsize=30)
    plt.ylabel("Loss", fontsize=30)

    # Add a legend and show the chart
    plt.legend(fontsize=30)
    plt.show()
