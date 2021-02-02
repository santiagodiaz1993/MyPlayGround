import numpy as np

# Passed in gradient from the next layer fro the pupose of this example we are
# going to use an array of an incremental gradient values
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
# We have 3 sets of inpurts - samples
inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs this 4 weights recall that we keep weights transposed
weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
).T

# One bias for each neuron biases are the row vector with shape (1, neuron)
biases = np.array([[2, 3, 0.5]])

# foward pass
layer_ouputs = np.dot(inputs, weights) + biases  # dense layer
relu_ouputs = np.maximum(0, layer_ouputs)  # ReLu activation

# Lets optimize and test backpropagation here. ReLu activation - simulates
# derivative with respect to input values from the next layer passed to current
# layer during backpropagation
drelu = relu_ouputs.copy()
drelu[layer_ouputs <= 0] = 0

# Danese layer
# inputs - multivariable by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)

# dbiases - sum values, do this over samples (first axis), keepdims since this
# by default will produce a plain list - we explained this in the chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)

# update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(dbiases)
