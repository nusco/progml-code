# Plot the sigmoid function.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


MARGIN_LEFT = -10
MARGIN_RIGHT = 10

# Configure axes
plt.axis([MARGIN_LEFT, MARGIN_RIGHT, -0.5, 1.5])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot margins
plt.plot([MARGIN_LEFT, MARGIN_RIGHT], [1, 1], color="grey", linestyle="dashed")
plt.plot([MARGIN_LEFT, MARGIN_RIGHT], [0, 0], color="grey", linestyle="dashed")

# Plot sigmoid
X = np.linspace(MARGIN_LEFT, MARGIN_RIGHT, 1000)
Y = [sigmoid(x) for x in X]
plt.plot(X, Y, color="blue")

plt.show()
