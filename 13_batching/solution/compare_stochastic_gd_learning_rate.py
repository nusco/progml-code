# Compare neural network performance with different learning rates, using
# stochastic GD.

import numpy as np
import mnist
import compare as cmp

BATCH_SIZE = 35      # Use stochastic GD
TIME = 1200          # Run each experiment for 20 minutes
HIDDEN_NODES = 200   # The number of hidden nodes

np.random.seed(1234)

cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=BATCH_SIZE, lr=0.1,
              time_in_seconds=TIME,
              label="lr = 0.1",
              color='orange', linestyle='-')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=BATCH_SIZE, lr=0.01,
              time_in_seconds=TIME,
              label="lr = 0.01",
              color='green', linestyle='--')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=BATCH_SIZE, lr=0.001,
              time_in_seconds=TIME,
              label="lr = 0.001",
              color='blue', linestyle='-.')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=BATCH_SIZE, lr=0.0001,
              time_in_seconds=TIME,
              label="lr = 0.0001",
              color='black', linestyle=':')

cmp.show_plot()
