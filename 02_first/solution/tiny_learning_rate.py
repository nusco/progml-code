import numpy as np
import linear_regression

# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system with a learning rate of 0.00001
w, b = linear_regression.train(X, Y, iterations=10000, lr=0.00001)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, linear_regression.predict(20, w, b)))
