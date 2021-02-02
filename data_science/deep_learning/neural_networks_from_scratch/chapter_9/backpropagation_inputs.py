import numpy as np

# Passed in gradiatn fron the next layer for the purpose of this example we are
# going to use a vector of ones
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# We have 3 sets of weights - one set for each neuron. We have 4 inputs, thus
# we have 4 weights. Recall that we keep weights transposed
weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
).T

dinputs = np.dot(dvalues, weights.T)

print(dinputs)
