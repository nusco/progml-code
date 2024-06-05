# Plot the reservations/pizzas dataset.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()                                                # activate Seaborn
plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
plt.xticks(fontsize=15)                                  # set x axis ticks
plt.yticks(fontsize=15)                                  # set y axis ticks
plt.xlabel("Reservations", fontsize=30)                  # set x axis label
plt.ylabel("Pizzas", fontsize=30)                        # set y axis label
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)  # load data
plt.plot(X, Y, "bo")                                     # plot data
plt.show()                                               # display chart
