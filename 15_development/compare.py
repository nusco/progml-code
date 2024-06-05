# Utility functions to compare different configurations of the network.
# For a concrete example of how to use this, look at compare_batch_sizes.py.

import neural_network as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns


# This train() replaces the one in neural_network.py, and it differs from that
# one in a few ways:
#
# * it runs for a specified time, rather than a specified number of epochs;
# * it runs quietly instead of reporting the loss and accuracy at each step;
# * at each step it stores the loss and time passed, and finally returns those
#   histories to the caller;
# * it also returns the number of training epochs and the total number of
#   gradient descent steps.
def train(X_train, Y_train,
          X_validation, Y_validation,
          n_hidden_nodes, lr, batch_size, time_in_seconds):
    n_input_variables = X_train.shape[1]
    n_classes = Y_train.shape[1]

    w1, w2 = nn.initialize_weights(n_input_variables,
                                   n_hidden_nodes,
                                   n_classes)
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


# Train the network with the specified configuration
def configuration(data, n_hidden_nodes, batch_size, lr,
                  time_in_seconds, label, color, linestyle):
    print("Training:", label)
    times, losses, epochs, steps = train(data.X_train, data.Y_train,
                                         data.X_validation, data.Y_validation,
                                         n_hidden_nodes=n_hidden_nodes,
                                         batch_size=batch_size,
                                         lr=lr,
                                         time_in_seconds=time_in_seconds)
    print("  Loss: %.8f (%d epochs completed, %d total steps)" %
          (losses[-1], epochs, steps))
    plt.plot(times, losses, label=label, color=color, linestyle=linestyle)


# Show a chart comparing the loss histories for all configurations
def show_results():
    sns.set()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Seconds", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.legend(fontsize=30)
    plt.show()
