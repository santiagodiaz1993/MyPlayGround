import numpy as np

# Passed in gradiatn fron the next layer for the purpose of this example we are
# going to use a vector of ones
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# We have 3 sets of weights - one set for each neuron. We have 4 inputs, thus
# we have 4 weights. Recall that we keep weights transposed
weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
).T

inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

# Sum wegiths related to the given input multiplied by the gradient related to
# the given neuron
# This is an "inenficient way" to do this. This is being done manually but
# below is done using numpy with a single line of code that does a dot product
dx0 = sum(
    [
        weights[0][0] * dvalues[0][0],
        weights[0][1] * dvalues[0][1],
        weights[0][2] * dvalues[0][2],
    ]
)

dx0 = sum(
    [
        weights[1][0] * dvalues[0][0],
        weights[1][1] * dvalues[0][1],
        weights[2][2] * dvalues[0][2],
    ]
)

dx0 = sum(
    [
        weights[2][0] * dvalues[0][0],
        weights[2][1] * dvalues[0][1],
        weights[2][2] * dvalues[0][2],
    ]
)

dx0 = sum(
    [
        weights[3][0] * dvalues[0][0],
        weights[3][1] * dvalues[0][1],
        weights[3][2] * dvalues[0][2],
    ]
)

# We can simplify this if we do the dot product
dinputs = np.dot(dvalues, weights.T)
dweights = np.dot(inputs.T, dvalues)
print("These is the derivative with respect to the input")
print(dinputs)
print("These is the derivative with respect to the weights")
print(dweights)

# One bias for each neuron biases are the row vector with shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# dbiases - sum values do this over samples (first axis), keepdims we explained
# this in chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)
