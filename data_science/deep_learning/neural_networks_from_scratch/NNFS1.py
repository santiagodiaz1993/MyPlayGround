import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

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


# Sigmoid activation
class Activation_Sigmoid:
    # Foward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivatives - calculate from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output


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


class Loss_BinaryCrossentropy(Loss):
    # foward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by zero
        # Clip both sides to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample wise loss
        sample_losses = -(
            y_true * np.log(y_pred_clipped)
            + (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        sample_losses = np.mean(sample_losses, axis=1)

        return sample_losses

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # Number of ouput in every sample
        # Well use the first sample to count him
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = (
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))
            / outputs
        )

        # Normalize gradients
        self.dinputs = self.dinputs / samples


class Activation_Linear:

    # forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.outputs = inputs

    # Backward pass
    def backward(self, dvalues):
        # Derivative is 1, 1 * dvalues = dvalues - chain rule
        self.dinputs = dvalues.copy()


class Loss_MeanSquaredError(Loss):

    # forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=1)
        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of ouputs in every sample
        # Well use the first sample to count them
        ouputs = len(dvalues[0])

        # Gradients on values
        self.dinputs = -2 * (y_true - dvalues) / ouputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):  # L1 loss
    def forward(self, y_pred, y_true):
        # calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=1)

        # return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of outputs in every sample
        # well use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


#  Create data set
# X, y = spiral_data(samples=100, classes=2)
#
# # Reshape lables to be a list of lists
# # Inner lists contains one ouput (either 0 or 1)
# # per each ouput neuron, 1 in this class
# y = y.reshape(-1, 1)
#
# dense1 = Layer_Dense(
#     2, 64, weights_regularization_l2=5e-4, bias_regularizer_l2=5e-4
# )
#
# activation1 = Activation_ReLu()
# # create the testing data
#
# dense2 = Layer_Dense(64, 1)
#
# activation2 = Activation_Sigmoid()
#
# loss_function = Loss_BinaryCrossentropy()
#
# optimizer = Optimizer_Adam(decay=5e-7)
#
#
# # Trainning in the loop
# for epoch in range(10001):
#
#     # Perform a foward pass of our trainning data through this layer
#     dense1.forward(X)
#
#     # Perform a foward pass through activation function takes the output of
#     # first dense layer here
#     activation1.forward(dense1.output)
#
#     # Perform a foward pass through the second dense layer takes outputs of
#     # second layer and reurns loss
#     dense2.forward(activation1.output)
#
#     activation2.forward(dense2.output)
#
#     # calculate loss from ouput of activation2 so softmax activation
#     # print("activation2 and y are both nd.arrays")
#     # print(type(activation2.output))
#     # print(type(y))
#     data_loss = loss_function.calculate(activation2.output, y)
#
#     # calculated regularization penalty
#     regularization_loss = loss_function.regularization_loss(
#         dense1
#     ) + loss_function.regularization_loss(dense2)
#
#     # Perform a foward pass through the activation/loss function takes the
#     # ouput of second dense layer here and returns loss
#     loss = data_loss + regularization_loss
#
#     # Calculate accuracy from output of the activation2 and targets
#     # calculate values along first axis
#     predictions = (activation2.output > 0.5) * 1
#     accuracy = np.mean(predictions == y)
#
#     # print(type(loss))
#     # print(type(epoch))
#     # print(type(regularization_loss))
#     # print(type(optimizer.current_learning_rate))
#     # loss2 = {loss: .3f}
#     if not epoch % 100:
#         print(
#             f"epoch: {epoch}, "
#             + f"acc: {accuracy:.3f}, "
#             + f"loss: {loss:.3f} ("
#             + f"data_loss: {data_loss:.3f}, "
#             + f"reg_loss: {regularization_loss:.3f}), "
#             + f"lr: {optimizer.current_learning_rate}"
#         )
#
#     # Backward pass
#     loss_function.backward(activation2.output, y)
#     activation2.backward(loss_function.dinputs)
#     dense2.backward(activation2.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)
#
#     # Update the weights and biases
#     optimizer.pre_update_params()
#     optimizer.update_params(dense1)
#     optimizer.update_params(dense2)
#     optimizer.post_update_params()
#
# # Validate model
# # Create test datasets
# X_test, y_test = spiral_data(samples=100, classes=2)
#
# # Rreshape labels to be a list of lists
# # Inner list contains one ouput (either 0 or 1)
# # Per each output neuron, 1 in this case
# y_test = y_test.reshape(-1, 1)
#
# # Perform a foward pass for our testing data through this layer
# dense1.forward(X_test)
#
# # Perform a forward pass through activation function
# # takes the output of first dense layer here
# activation1.forward(dense1.output)
#
# # Perform a forward pass through second Dense Layer
# # takes the output of the activation function of first layer as inputs
# dense2.forward(activation1.output)
#
# # Perform a forward pass through activation function
# # takes the output of second dense layer here
# activation2.forward(dense2.output)
#
# # Calculate the data loss
# loss = loss_function.calculate(activation2.output, y_test)
#
# # Calculate accuracy from output of activation2 and targets
# # part in brackets returns a binary mask array consisting of
# # True / False values, multiplying it by 1 changes it into array
# # of 1s and 0s
# predictions = (activation2.output > 0.5) * 1
# accuracy = np.mean(predictions == y_test)
#
# print(f"validation, acc: {accuracy: .3f}, loss : {loss: .3f}")
#
##############################################################################
# Regrestion testing
X, y = sine_data()

# Create Dense Layer with 1 input feature and 64 output values
dense1 = Layer_Dense(1, 64)

# Create ReLU activation (To be used with Dense Layer ):
activation1 = Activation_ReLu()

# Create second Dense Layer with 64 input features (as we take ouput
# of previous layer here) and 1 output value
dense2 = Layer_Dense(64, 1)

# Create Linear activation:
activation2 = Activation_Linear()

# Create loss function
loss_function = Loss_MeanSquaredError()

# Create optimizer
optimizer = Optimizer_Adam()

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. Well calculate it by checking how many
# values have a difference to their graoudn thruth equivalent less than given
# precison.
# We'll calculate this precision as a fraction of standar deviation of al the
# ground thruth values
accurate_precision = np.std(y) / 250

# Train in loop
for epoch in range(1001):
    # Perform a forward pass of our trainning
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the ouput of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second dense layer takes ouputs of
    # activation  function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a foward pass through activation function takes the ouput of
    # second dense layer here
    activation2.forward(dense2.output)

    # Calculate the data loss
    data_loss = loss_function.calculate(activation2.outputs, y)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(
        dense1
    ) + loss_function.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from ouput of activation2 and targets
    # to calculate it we're talking absolute difference between prediction and
    # ground truth values and compare if difference are lower than given
    # precision value
    predictions = activation2.outputs
    accuracy = np.mean(np.absolute(predictions - y) < accurate_precision)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy:.3f}, "
            + f"loss: {loss:.3f} ("
            + f"data_loss: {data_loss:.3f}, "
            + f"reg_loss: {regularization_loss:.3f}), "
            + f"lr: {optimizer.current_learning_rate}"
        )

    # Backward pass
    loss_function.backward(activation2.outputs, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
