# Compare the network's accuracy with different learning rates.

import compare
import mnist_standardized

DATA = mnist_standardized
HIDDEN = 100
BATCH = 128
TIME = 60 * 10
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=BATCH,
                      lr=0.001, time_in_seconds=TIME,
                      label="lr=0.001", color='orange', linestyle=':')
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=BATCH,
                      lr=0.01, time_in_seconds=TIME,
                      label="lr=0.01", color='green', linestyle='-.')
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=BATCH,
                      lr=0.1, time_in_seconds=TIME,
                      label="lr=0.1", color='blue', linestyle='--')
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=BATCH,
                      lr=1, time_in_seconds=TIME,
                      label="lr=1", color='black', linestyle='-')
compare.show_results()
