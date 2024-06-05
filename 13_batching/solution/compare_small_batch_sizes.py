# Compare neural network performance with different batch sizes.

import numpy as np
import mnist
import compare as cmp

TIME = 1200         # Run each experiment for 20 minutes
HIDDEN_NODES = 200  # The number of hidden nodes
lr = 0.01           # The learning rate

np.random.seed(1234)

cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=mnist.X_train.shape[0], lr=lr,
              time_in_seconds=TIME,
              label="Batch GD",
              color='orange', linestyle='-')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=3, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 3",
              color='green', linestyle='--')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=5, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 5",
              color='blue', linestyle='-.')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=7, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 7",
              color='black', linestyle=':')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=12, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 12",
              color='red', linestyle='--')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=24, lr=lr,
              time_in_seconds=TIME,
              label="Batch size 15",
              color='cyan', linestyle=':')

cmp.show_plot()
