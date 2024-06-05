# Plot the loss on a dataset with a single input variable.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


# Load data
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

sns.set()  # Activate Seaborn

# Compute losses for w ranging from -1 to 4
weights = np.linspace(-1.0, 4.0, 200)
losses = [loss(X, Y, w, 0) for w in weights]

# Plot weights and losses
plt.axis([-1, 4, 0, 1000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Weight", fontsize=30)
plt.ylabel("Loss", fontsize=30)
plt.plot(weights, losses, color="black")

# Put a green cross on the minimum loss
min_index = np.argmin(losses)
plt.plot(weights[min_index], losses[min_index], "gX", markersize=26)

plt.show()
