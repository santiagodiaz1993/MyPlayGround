import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import random

nnfs.init()


# Desnse layer
class Layer_Dense:

    # Initialize weights and biases
    def __init__(
        self,
        n_inputs,
        n_neurons,
        weights_regularization_l1=0,
        weights_regularization_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularizsation strength
        self.weights_regularization_l1 = weights_regularization_l1
        self.weights_regularization_l2 = weights_regularization_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Foward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # calculate output values from inputs wights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # Gradients on parameter
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weights_regularization_l1 > 0:
            dL1 = np.one_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weights_regularization_l1 * dL1
        # L2 on weights
        if self.weights_regularization_l2 < 0:
            self.dweights += 2 * self.weights_regularization_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.01 we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs):
        # Save inputs values
        self.inputs = inputs
        # Genera and save scaled masks
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.inputs = dvalues * self.binary_mask


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
        # Since we need to modify the original variable, lets make a copy of
        # the values first
        self.dinputs = dvalues.copy()

        # zero gradient where inputs value were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation function
class Activation_Softmax:

    # Foward pass
    def forward(self, inputs):
        # Remember inputs values
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


# SGBD optimizer
class Optimizer_SGBD:
    # Initializing optimizer - set settings
    # Learning rate for 1, is the default for this optimizer
    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any paratmeter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1 + self.decay * self.iterations)
            )

    # Update paramters
    def update_params(self, layer):

        # If momentum has been set
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled
            # with zeros
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array does not exit for biases yet either
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and updates with
            # current gradient
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weights_momentums = weight_updates

            # build biases updates
            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates

        # Vanilla SGF updates (without momentum, only LR, and decay)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


# Adagrad optimizer
class Optimizer_Adagrad:

    # Initializing optimizer - set settings
    def __init__(self, learning_rate=1, decay=0, epsilon=0.0000001):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any paratmeter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1 + self.decay * self.iterations)
            )

    # Update paramters
    def update_params(self, layer):

        # if layer does not contrain cache arrays
        # create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with square current gradients
        layer.weight_cache += layer.weights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with saquare rooted cache
        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1 + self.decay * self.iterations)
            )

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (
            self.rho * layer.weight_cache
            + (1 - self.rho) * layer.dweights ** 2
        )
        layer.bias_cache = (
            self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        )

        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(
        self,
        learning_rate=0.001,
        decay=0,
        epsilon=1e-7,
        beta_1=0.9,
        beta_2=0.999,
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1 + self.decay * self.iterations)
            )

    # update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # craete them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums
            + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums
            + (1 - self.beta_1) * layer.dbiases
        )

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache
            + (1 - self.beta_2) * layer.dweights ** 2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache
            + (1 - self.beta_2) * layer.dbiases ** 2
        )
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        # Vanilla SGD paratmeter update + normalization
        # with square rooted cache
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


# Common loss class
class Loss:
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        # calculate only when the factor is greater than 0
        if layer.weights_regularization_l1 > 0:
            regularization_loss += layer.weights_regularization_l1 * np.sum(
                np.abs(layer.weights)
            )

        # L2 regularization - weights
        # calculate only when the factor is greater than 0
        if layer.weights_regularization_l1 > 0:
            regularization_loss += layer.weights_regularization_l2 * np.sum(
                np.sum(layer.weights * layer.weights)
            )

        # L1 regularization - bias
        # calculate only when the factor is greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(
                np.abs(layer.biases)
            )

        # L1 regularization - bias
        # calculate only when the factor is greater than 0
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(
                np.sum(layer.biases * layer.biases)
            )

        return regularization_loss

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
    def forward(self, inputs, y_true):

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


#  Create data set
X, y = spiral_data(samples=100, classes=3)

# create the testing data
X_test, y_test = spiral_data(samples=100, classes=3)

# Create Dense Layer with 2 inputs features and 64 inputs values
dense1 = Layer_Dense(
    2, 64, weights_regularization_l2=5e-4, bias_regularizer_l2=5e-4
)

# Create ReLu activation (to be used with Dense layer)
activation1 = Activation_ReLu()

# Create drop out layer
dropout1 = Layer_Dropout(0.1)

# Create second Dense Layer with 64 inputs features (as we take output of
# previous layer here) and 3 ouput values (ouput values)
dense2 = Layer_Dense(64, 3)

# Create softmax classifier combined loss activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
# optimizer = Optimizer_SGBD(decay=1e-3, momentum=0.5)
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-4, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)


# print("weights after creating uptimizer")
# print(dense1.weights)
for epoch in range(10001):

    # Perform a foward pass of our trainning data through this layer
    dense1.forward(X_test)

    # Perform a foward pass through activation function takes the output of
    # first dense layer here
    activation1.forward(dense1.output)

    # Perform a foward pass through Dropout Layer
    dropout1.forward(activation1.output)

    # Perform a foward pass through the second dense layer takes outputs of
    # second layer and reurns loss
    dense2.forward(activation1.output)

    # calculate loss from ouput of activation2 so softmax activation
    data_loss = loss_activation.forward(dense2.output, y)

    # calculated regularization penalty
    regularization_loss = loss_activation.loss.regularization_loss(
        dense1
    ) + loss_activation.loss.regularization_loss(dense2)

    # Perform a foward pass through the activation/loss function takes the
    # ouput of second dense layer here and returns loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of the activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)

    accuracy = np.mean(predictions == y_test)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy: .3f}, "
            + f"loss: {loss:.3f}, "
            + f"data_loss: {data_loss:.3f}, "
            + f"reg_loss: {regularization_loss:.3f}, "
            + f"lr:  {optimizer.current_learning_rate}",
        )

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.inputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
