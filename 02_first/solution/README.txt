This folder contains two experiments. One runs linear regression with a very
large learning rate (big_learning_rate.py), and the other with a very small one
(tiny_learning_rate.py).

With a learning rate of 1, the algorithm takes just a handful of iterations to
find a result, but that quick result is unlikely to generate reliable
predictions. At each iteration, the line moves in large 1-sized steps, and
doesn't have the finesse to approximate the points very accurately. In fact, if
you generate a few predictions for the dataset in pizza.txt, you'll see that
they tend to miss the mark pretty badly.

With a learning rate of 0.00001, the line moves in tiny baby steps. The
training comes up with a precise estimate for W and B, but it takes a long time
to get there. In fact, it doesn't even converge within the allotted 10000
iterations. If you try giving it more iterations, you'll see that it takes over
1.5 million iterations to find a line. That's a long time to wait for such a
simple algorithm!

So, to recap: a small learning rate increases precision at the expense of
training speed. I didn't show you how to measure the accuracy of predictions
yet, so you'll have to trust your gut feeling when you pick the learning rate,
or try a few random values. In my examples, I settled on a learning rate of
0.01, that looks more than precise enough for the problem at hand, and makes
for reasonably fast training.
