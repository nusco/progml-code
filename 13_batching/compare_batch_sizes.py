# Compare neural network performance with different batch sizes,
# from stochastic GD (batch size 1) to batch GD (one large batch).

import numpy as np
import mnist
import compare as cmp

TIME = 1800         # The running time of each batch size, in seconds
HIDDEN_NODES = 200  # The number of hidden nodes
lr = 0.01           # The learning rate

np.random.seed(1234)

cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=1, lr=lr,
              time_in_seconds=TIME,
              label="Stochastic GD",
              color='orange', linestyle='-')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=32, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 32",
              color='green', linestyle='--')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=128, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 128",
              color='blue', linestyle='-.')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=mnist.X_train.shape[0], lr=lr,
              time_in_seconds=TIME,
              label="Batch GD",
              color='black', linestyle=':')

cmp.show_plot()
