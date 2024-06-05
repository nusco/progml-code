# Compare the network's accuracy on MNIST with and without standardization.

import neural_network as nn
import mnist as normal
import mnist_standardized as standardized

print("Regular MNIST:")
nn.train(normal.X_train, normal.Y_train,
         normal.X_validation, normal.Y_validation,
         n_hidden_nodes=200, epochs=2, batch_size=60, lr=0.1)

print("Standardized MNIST:")
nn.train(standardized.X_train, standardized.Y_train,
         standardized.X_validation, standardized.Y_validation,
         n_hidden_nodes=200, epochs=2, batch_size=60, lr=0.1)
