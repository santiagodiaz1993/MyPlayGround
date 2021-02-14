import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from timeit import timeit

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

        self.inputs = inputs

    def backward(self, dvalues):
        # Gradients on parameter
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# RElu activation
class Activation_ReLu:

    # Foward pass
    def forward(self, inputs):
        # Remember inputs values
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        # since we need to modify the original variable, lets make a copy of
        # the values first
        self.dinputs = dvalues.copy()

        # zero gradient where inputs value were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    # Foward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumarate outputs and gradients
        for index, (single_output, single_dvalue) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calvulate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            # Calculate sample-wise gradient and add it to the array of sample
            # gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)


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

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of labels in every sample well use the first sample to count
        # them
        labels = len(dvalues[0])

        # if lables are sparse, trun them into one-hot vector
        # check for encoding, if discrete or one hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate the gradient
        self.dinputs = -y_true / dvalues  # Target values / predicted values

        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined softmax activation and cross-entropy loss for
# faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Foward pass
    def foward(self, inputs, y_true):

        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the ouput
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # print("y_ture is")
        # print(y_true)
        # print("dvalues are")
        # print(dvalues)

        # Number of sample
        samples = len(dvalues)
        # print("these are the samples")
        # print(samples)

        # if labels are one-hot encoded, turn them into descrete values
        if len(y_true.shape) == 2:
            # print("Len is longer than 2 and it is:")
            # print(y_true.shape)
            y_true = np.argmax(y_true, axis=1)
            # print("and now we have turned it into")
            # print(y_true)

        # Copy so we can safeley modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # print("the derivative of the inputs are ")
        # print(self.dinputs)
        # Normalize gradient
        self.inputs = self.dinputs / samples
        # print("the inputs are ")
        # print(self.inputs)


# Create dataset
x, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 inputs features and 3 outputs values
dense1 = Layer_Dense(2, 3)

# Create ReLu activation  (to be used with Dense layer ):
activation1 = Activation_ReLu()

# Create second  Dense layer with 3 inputs features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create softmax classifiers combined loss activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Make a forward pass of our tranning data through this layer
dense1.forward(x)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes the output of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a foward pass throuugh the activation/loss function takes the ouput
# of second dense layer here and returns loss
loss = loss_activation.foward(dense2.output, y)

# Les see output of the first few examples
print(loss_activation.output[:5])

# print loss value
print("This is the loss", loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

# print accuracy
print("accuracy", accuracy)

# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
# print("this are the dinputs from dense2")
# print(dense2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# print gradients
# print(dense1.weights)
# print(dense1.dbiases)
# print(dense2.weights)
# print(dense2.dbiases)


class speedtestsoftmaxpluscategoricalloss:
    softmax_ouputs = np.array(
        [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
    )

    class_targets = np.array([0, 1, 1])

    def f1(softmax_ouputs, class_targets):
        softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
        softmax_loss.backward(softmax_ouputs, class_targets)
        dvalues1 = softmax_loss.inputs
        print("Gradients: combined loss and activation:")
        print(dvalues1)

    def f2(softmax_ouputs, class_targets):
        activation = Activation_Softmax()
        activation.output = softmax_ouputs
        loss = Loss_CategoricalCrossentropy()
        loss.backward(softmax_ouputs, class_targets)
        activation.backward(loss.dinputs)
        dvalues2 = activation.dinputs
        print("Gradients: seperate loss and activation:")
        print(dvalues2)

    # TODO(sdiaz): Fix the testing of the speed
    # t1 = timeit(lambda: f1(softmax_ouputs, class_targets), number=10000)
    # t2 = timeit(lambda: f2(softmax_ouputs, class_targets), number=10000)

    # print(t1 / t2)
