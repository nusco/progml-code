Here is the problem that I hinted at: the data could be ordered. For example, imagine what would happen if the characters dataset contained all the 'A's first, followed by all the 'B's, and so on until the 'Z's. If that happens (and it often does), then splitting the dataset would get us three sets with a different distribution: some letters, such as 'A's, could only be found in the training set, while others, such as 'Z's, would all cluster in the test set. During training, the system would never get a chance to learn what a 'Z' looks like. Also, the system's accuracy metrics on the validation and the test set would be quite unreliable.

This problem might be obvious for something like alphabetically sorted labels, but it could take much sneakier forms. For example, imagine how hard it could be to notice that a dataset of musical tracks tends to have most folk songs in the beginning, or that a dataset of cat pictures clusters all sleeping cats near the end.

In general, we should take care that the distribution of the examples is the same across the following four sets of data:

- the training set;
- the validation set;
- the test set;
- the actual data that the system classifies in production.

While the production data can be challenging to deal with, there's an easy fix to make it more likely that the first three sets have a similar distribution: shuffle the data before you split it into training, validation and test sets.

    np.random.shuffle(data_all)
    data_train, data_validation, data_test = np.split(data_all, [900_000, 950_000])

MNIST already comes pre-shuffled. That's why we can just go ahead, and split the test set in the middle to get a validation set.
