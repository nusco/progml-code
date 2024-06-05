import neural_network as nn
import mnist_standardized as data

nn.train(data.X_train, data.Y_train, data.X_test, data.Y_test,
         n_hidden_nodes=100, epochs=10, batch_size=256, lr=1)
