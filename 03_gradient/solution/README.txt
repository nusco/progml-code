In loss_increasing.py, I set a learning rate of 0.005, instead of the 0.001 I
used in the book. That causes the loss to increase at each iteration:

Iteration    0 => Loss: 812.8666666667
Iteration    1 => Loss: 1131.8941261333
Iteration    2 => Loss: 1587.8389543973
...

To understand why the loss increases like that, check out the picture in this
folder. (It's a sneak peek at a more detailed explanation that will come later
in the book). The learning rate measures how large each step of gradient
descent is. If those steps are too large, it's possible that each step will
overshoot the minimum and end on a peak that's even higher than the starting
point. That's what happens with gradient descent when the hiker drinks too much
coffee.
