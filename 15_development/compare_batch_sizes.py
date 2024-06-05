# Compare the network's accuracy with different batch sizes.

import compare
import mnist_standardized

DATA = mnist_standardized
HIDDEN = 100
LR = 1
TIME = 60 * 5
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=60000,
                     lr=LR, time_in_seconds=TIME,
                     label="batch_size=60000", color='orange', linestyle=':')
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=256,
                     lr=LR, time_in_seconds=TIME,
                     label="batch_size=256", color='green', linestyle='-.')
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=128,
                     lr=LR, time_in_seconds=TIME,
                     label="batch_size=128", color='blue', linestyle='--')
compare.configuration(data=DATA, n_hidden_nodes=HIDDEN, batch_size=64,
                     lr=LR, time_in_seconds=TIME,
                     label="batch_size=64", color='black', linestyle='-')
compare.show_results()
