# Compare neural network performance with batch GD vs. mini-batch GD

import numpy as np
import mnist
import compare as cmp

TIME = 1200         # The running time of each batch size, in seconds
HIDDEN_NODES = 200  # The number of hidden nodes
lr = 0.01           # The learning rate

np.random.seed(1234)

cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=mnist.X_train.shape[0], lr=lr,
              time_in_seconds=TIME,
              label="Batch GD",
              color='black', linestyle='-')
cmp.plot_loss(data=mnist, n_hidden_nodes=HIDDEN_NODES,
              batch_size=256, lr=lr,
              time_in_seconds=TIME,
              label="Mini-batch GD",
              color='green', linestyle='--')

cmp.show_plot()
