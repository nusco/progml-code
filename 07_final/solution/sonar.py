# Load and prepare Sonar data. For a detailed description of the Sonar
# dataset, see: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

# Load the entire dataset into a large (208, 61) NumPy array:
import csv
import numpy as np
data_as_python_list = list(csv.reader(open("sonar.all-data")))
data_without_bias_column = np.array(data_as_python_list)

# Prepend a bias column, resulting in a (208, 62) matrix:
data = np.insert(data_without_bias_column, 0, 1, axis=1)

# Shuffle data. This is important, because the Sonar dataset contains all
# "Rock" examples first, and all "Metal" examples later. If we don't shuffle it,
# we'll end up with a test set composed exclusively of "Metal" examples when
# we split the dataset later.
np.random.seed(1234)      # Have the same predictable shuffle every time
np.random.shuffle(data)   # Shuffle matrix rows in place

# Extract a (208, 61) input matrix:
#   - [:, 0:-1] stands for: "all rows, all columns except the last one"
#   - Convert all strings to float
X = data[:, 0:-1].astype(np.float32)

# Extract a (208, 1) matrix of labels:
#   - [:, -1] stands for: "extract all rows, but only the last column"
#   - Reshape to be 1 column and as many rows as necessary
#   - Convert all 'M's to True and all 'R's to False
#   - Convert True and False to 1 and 0, respectively
labels = data[:, -1].reshape(-1, 1)
Y_unencoded = (labels == 'M').astype(np.int)

# Split into training and test set:
SIZE_OF_TRAINING_SET = 160  # Keep the remaining 48 elements for testing
X_train, X_test = np.vsplit(X, [SIZE_OF_TRAINING_SET])
Y_train_unencoded, Y_test = np.vsplit(Y_unencoded, [SIZE_OF_TRAINING_SET])

# One hot encode the training set, but not the test set:
def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 2
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

Y_train = one_hot_encode(Y_train_unencoded)
