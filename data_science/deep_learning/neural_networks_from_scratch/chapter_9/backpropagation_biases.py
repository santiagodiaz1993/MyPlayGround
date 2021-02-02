import numpy as np

# Passed in gradiatn fron the next layer for the purpose of this example we are
# going to use a vector of ones
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# We have 3 sets of weights - one set for each neuron. We have 4 inputs, thus
# we have 4 weights. Recall that we keep weights transposed
biases = np.array([[2, 3, 0.5]])

# dbiases - sum values, do this over samples (first axis), keepdims, we
# explained this in chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)

print(dbiases)
