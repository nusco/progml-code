I didn't try to improve this network further after writing this chapter, so the best hyperparameter configuration I got is the one that I showed you in the book. It has 95% accuracy on the training set, and over 92% on the validation set:

  loss: 0.1684 - acc: 0.9509 - val_loss: 0.2278 - val_acc: 0.9263

In this solution, I replaced the validation set with the test set. I marked the few lines I changed in the neural network's code with comments. The other files didn't change.

When I ran this final test, I got:

  loss: 0.1684 - acc: 0.9509 - val_loss: 0.2017 - val_acc: 0.9333

I was a bit concerned that the system would overfit the validation set, because that's the set I used to tune my regularization hyperparameters. Apparently, that didn't happen. In fact, the system's accuracy on the test set is even a touch higher than the accuracy on the validation set. That's a nice surprise!

If you did your own tuning of the network's hyperparameters, then you can run this same test on your own regularized network.
