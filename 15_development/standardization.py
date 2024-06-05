# An example of standardized data.

import numpy as np

inputs = [1, 2, 3, 4, 5, 6, -10]

print("Inputs:", np.average(inputs))
print("Inputs average:", np.average(inputs))
print("Inputs standard deviation:", np.std(inputs))

standardized_inputs = (inputs - np.average(inputs)) / np.std(inputs)

# The standardized inputs average might not be exactly zero, because
# of rounding errors.
print("Standardized inputs:", np.average(standardized_inputs))
print("Standardized inputs average:", np.average(standardized_inputs))
print("Standardized inputs standard deviation:", np.std(standardized_inputs))
