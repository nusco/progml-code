# Compare the network's accuracy with different numbers of hidden nodes.

import compare
import mnist_standardized

DATA = mnist_standardized
BATCH = 128
LR = 0.1
TIME = 60 * 10
compare.configuration(data=DATA, n_hidden_nodes=10, batch_size=BATCH,
                      lr=LR, time_in_seconds=TIME,
                      label="h=10", color='orange', linestyle='-')
compare.configuration(data=DATA, n_hidden_nodes=100, batch_size=BATCH,
                      lr=LR, time_in_seconds=TIME,
                      label="h=100", color='green', linestyle='--')
compare.configuration(data=DATA, n_hidden_nodes=400, batch_size=BATCH,
                      lr=LR, time_in_seconds=TIME,
                      label="h=400", color='blue', linestyle='-.')
compare.configuration(data=DATA, n_hidden_nodes=1000, batch_size=BATCH,
                      lr=LR, time_in_seconds=TIME,
                      label="h=1000", color='black', linestyle=':')
compare.show_results()
