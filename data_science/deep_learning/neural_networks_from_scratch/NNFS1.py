import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


# Desnse layer
class Layer_Dense:
    # Initialize weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calculate output values from inputs wights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# RElu activation
class Activation_ReLu:

    # Foward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    # Foward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create dataset
x, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 inputs features and 3 outputs values
dense1 = Layer_Dense(2, 3)

# Create ReLu activation  (to be used with Dense layer ):
activation1 = Activation_ReLu()

# Create second  Dense layer with 3 inputs features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our tranning data through this layer
dense1.forward(x)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)


# Make a forward pass through second Dense layer
# it takes the output of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of the second dense layer here
activation2.forward(dense2.output)

# Lets see the output of the first few samples
print(activation2.output[:5])
