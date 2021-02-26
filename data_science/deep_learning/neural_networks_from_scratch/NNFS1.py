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

    # Foward values
    def forward(self, inputs):
        print("these are the inputs")
        print(inputs)
        # Remember input values
        self.inputs = inputs
        # calculate output values from inputs wights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
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


# Softmax activation function
class Activation_Softmax:

    # Foward pass
    def forward(self, inputs):
        # Remember the inputs values
        self.inputs = inputs

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


# Common loss class
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
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

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

        # Number of sample
        samples = len(dvalues)

        # if labels are one-hot encoded, turn them into descrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safeley modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# SGBD optimizer
class Optimizer_SGBD:
    # Initializing optimizer - set settings
    # Learning rate for 1, is the default for this optimizer
    def __init__(self, learning_rate=1, decay=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any paratmeter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1 + self.decay * self.iterations)
            )

    # Update paramters
    def update_params(self, layer):
        # print("this is the current learning rate")
        # print(self.current_learning_rate)
        layer.weights += -self.current_learning_rate * layer.weights
        layer.biases += -self.current_learning_rate * layer.dbiases
        # print("this is the current layer weights")
        # print(layer.weights)

    def post_update_params(self):
        self.iterations += 1

    # x, y = spiral_data(samples=100, classes=3)
    #
    # # Create Dense layer with 2 inputs features and 3 outputs values
    # dense1 = Layer_Dense(2, 3)
    #
    # # Create ReLu activation  (to be used with Dense layer ):
    # activation1 = Activation_ReLu()
    #
    # # Create second  Dense layer with 3 inputs features (as we take output
    # # of previous layer here) and 3 output values
    # dense2 = Layer_Dense(3, 3)
    #
    # # Create softmax classifiers combined loss activation
    # loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    #
    # # Make a forward pass of our tranning data through this layer
    # dense1.forward(x)
    #
    # # Make a forward pass through activation function
    # # it takes the output of first dense layer here
    # activation1.forward(dense1.output)
    #
    # # Make a forward pass through second Dense layer
    # # it takes the output of activation function of first layer as inputs
    # dense2.forward(activation1.output)
    #
    # # Perform a foward pass throuugh the activation/loss function takes the ouput
    # # of second dense layer here and returns loss
    # loss = loss_activation.foward(dense2.output, y)
    #
    # # Les see output of the first few examples
    # print(loss_activation.output[:5])
    #
    # # print loss value


# print("This is the loss", loss)
#
# predictions = np.argmax(loss_activation.output, axis=1)
# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
# accuracy = np.mean(predictions == y)
#
# # Print accuracy
# print("accuracy", accuracy)
#
# # backward pass
# loss_activation.backward(loss_activation.output, y)
# dense2.backward(loss_activation.dinputs)
# activation1.backward(dense2.dinputs)
# dense1.backward(activation1.dinputs)
#
# print(dense1.dweights)
# print(dense1.dbiases)
# print(dense2.dweights)
# print(dense2.dbiases)
#
#
#  optimizer = Optimizer_SGBD()
#  optimizer.update_params(dense1)
#  optimizer.update_params(dense2)
#
#  Create data set

X, y = spiral_data(samples=100, classes=3)


# Create Dense Layer with 2 inputs features and 64 inputs values
dense1 = Layer_Dense(2, 64)

# Create ReLu activation (to be used with Dense layer)
activation1 = Activation_ReLu()

# Create second Dense Layer with 64 inputs features (as we take output of
# previous layer here) and 3 ouput values (ouput values)
dense2 = Layer_Dense(64, 3)

# Create softmax classifier combined loss activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_SGBD(decay=1e-2)

# print("weights after creating uptimizer")
# print(dense1.weights)
for epoch in range(10001):

    # Perform a foward pass of our trainning data through this layer
    print(10 * "#" + str(epoch))
    print("weights before doing a foward")
    print(dense1.weights)
    dense1.forward(X)
    print("weights after foward")
    print(dense1.weights)

    # Perform a foward pass through activation function takes the output of
    # first dense layer here
    activation1.forward(dense1.output)

    # Perform a foward pass through the second dense layer takes outputs of
    # second layer and reurns loss
    dense2.forward(activation1.output)

    # Perform a foward pass through the activation/loss function takes the
    # ouput of second dense layer here and returns loss
    loss = loss_activation.foward(dense2.output, y)

    # Calculate accuracy from output of the activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy: .3f}, "
            + f"loss: {loss:.3f}, "
            + f"lr:  {optimizer.current_learning_rate}",
        )

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# lass speedtestsoftmaxpluscategoricalloss:
#    softmax_ouputs = np.array(
#        [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
#    )
#
#    class_targets = np.array([0, 1, 1])
#
#    def f1(softmax_ouputs, class_targets):
#        softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
#        softmax_loss.backward(softmax_ouputs, class_targets)
#        dvalues1 = softmax_loss.inputs
#        print("Gradients: combined loss and activation:")
#        print(dvalues1)
#
#    def f2(softmax_ouputs, class_targets):
#        activation = Activation_Softmax()
#        activation.output = softmax_ouputs
#        loss = Loss_CategoricalCrossentropy()
#        loss.backward(softmax_ouputs, class_targets)
#        activation.backward(loss.dinputs)
#        dvalues2 = activation.dinputs
#        print("Gradients: seperate loss and activation:")
#        print(dvalues2)
#
#    # TODO(sdiaz): Fix the testing of the speed
#    # t1 = timeit(lambda: f1(softmax_ouputs, class_targets), number=10000)
#    # t2 = timeit(lambda: f2(softmax_ouputs, class_targets), number=10000)
#
#    # print(t1 / t2)
