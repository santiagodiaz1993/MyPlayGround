import numpy as np
import nnfs
from nnfs.datasets import vertical_data


nnfs.init()

# Desnse layer
class Layer_Dense:

    # Initialize weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calculate output values from inputs wights and biases
        print("this ware the inputs")
        print(inputs)
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


class Loss:

    # Calculating the data and regularization losses
    # given model output and ground thruth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Retrun loss
        return data_loss


# Cross-entropy loss
# this class is a child class from loss
class Loss_CategoricalCrossentropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0 (log(0) or log(1))
        # Clip both sides o not drag mean towards any value
        y_pred_cliped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_cliped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_cliped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


# create dataset
x, y = vertical_data(samples=100, classes=3)
print(x)
print(y)

# create model
dense1 = Layer_Dense(2, 3)  # First dense layer, 2 inputs
activation1 = Activation_ReLu()
dense2 = Layer_Dense(3, 3)  # Second layer, 3 inputs and 3 outputs
activation2 = Activation_Softmax()


# Create model
loss_function = Loss_CategoricalCrossentropy()

# saving weights and biases data into variables
lowest_loss = 99999  # some randome initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for iteration in range(10000):

    # Geneate a new set of eights and biases
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense1.weights = 0.05 * np.random.randn(3, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of the trainning data thoguh this layer
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a foward pass through activation function
    # it takes the ouput of second dense layer and returns los
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from outpout of activation2 and targets
    # calculate values along first axis
    predictions = np.argamx(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print(
            "New set of wights and biases found, iteration:",
            iteration,
            "loss",
            loss,
            "Accuracy",
            accuracy,
        )
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
