import numpy as np

# Passed in gradiant fron the next layer for the purpose of this example we are
# going to an use array of an incremental gradient values
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# We have three sets of inputs - Samples
inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

# Sum weights of given inputs and multiply by the passed-in gradient for this
# neuron
dweights = np.dot(inputs.T, dvalues)

print(dweights)
