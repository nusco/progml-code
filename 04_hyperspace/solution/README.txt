The file multiple_regression_final.py contains all the changes suggested in the
exercise's readme.txt. It loads the life expectancy data and trains the model
for 5 millions of iterations with a small learning rate of 0.0001.

The training took a few minutes on my computer. In the end, the predictions on
the first 10 countries look like these:

X[0] -> 75.5160 (label: 76)
X[1] -> 75.2524 (label: 74)
X[2] -> 77.4308 (label: 82)
X[3] -> 77.2738 (label: 81)
X[4] -> 70.1363 (label: 71)
X[5] -> 76.0045 (label: 75)
X[6] -> 73.4622 (label: 76)
X[7] -> 66.2005 (label: 71)
X[8] -> 76.3621 (label: 75)
X[9] -> 75.6892 (label: 72)

Some of the predictions are pretty close to the labels. Others miss the target
pretty badly. The reason for those failures is that our hyperplane-based model
is too simple to approximate the examples. When we introduced linear
regression, we said that you can approximate the points with a line only if the
points are roughly aligned to begin with. If you add more dimensions to the
points, the same reasoning applies. It seems that to approximate this complex
real-world dataset, we need a "non-straight" shape, instead of a "straight"
hyperplane.

In the last chapter of Part I ("The Perceptron") we'll talk more about this
limitation of multiple linear regression, and you'll see a few pictures that
will make it clearer.
