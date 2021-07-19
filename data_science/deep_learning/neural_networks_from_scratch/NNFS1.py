import numpy as np
import copy
import pickle
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
import os
import cv2

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
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularizsation strength
        self.weights_regularization_l1 = weights_regularization_l1
        self.weights_regularization_l2 = weights_regularization_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Foward pass
    def forward(self, inputs, training):
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

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Layer_Dropout:
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.01 we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs, training):
        # Save inputs values
        self.inputs = inputs

        # if not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Genera and save scaled masks
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:

    # Foward pass
    def forward(self, inputs, training):
        self.output = inputs


# RElu activation
class Activation_ReLu:

    # Foward pass
    def forward(self, inputs, training):
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

    # Calculate predictions for ouputs
    def predictions(self, outputs):
        return outputs


# Sigmoid activation
class Activation_Sigmoid:
    # Foward pass
    def forward(self, inputs, training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivatives - calculate from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Softmax actiavtion
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


# Softmax activation function
class Activation_Softmax:

    # Foward pass
    def forward(self, inputs, training):
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

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Softmax classifier - combined softmax activation and cross-entropy loss for
# faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Creates activation and loss function objects
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


class Activation_Linear:

    # forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # Derivative is 1, 1 * dvalues = dvalues - chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


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
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
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
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
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
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
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
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0

        # Calcualte regularization loss
        # iterate all trainable_layers
        for layer in self.trainable_layers:

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

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regulatization losses given model output and
    # ground truth values
    def calculate(self, output, y, *, include_regulatization=False):

        # calculate sample loss
        sample_loss = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_loss)

        self.accumulated_sum += np.sum(sample_loss)
        self.accumulated_count += len(sample_loss)

        # If just data loss - return it
        if not include_regulatization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Caulcate accumulated loss
    def calculate_accumulated(self, *, include_regulatization=False):

        # Calcualte mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regulatization:
            return data_loss

        # Return the data and regularization loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


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


class Loss_BinaryCrossentropy(Loss):
    # foward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by zero
        # Clip both sides to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample wise loss
        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
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
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        )

        # Normalize gradients
        self.dinputs = self.dinputs / samples


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


class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):
        # Create adn set the input layer
        self.input_layer = Layer_Input()
        # Count all objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If its the first layer
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # The last layer - the next object is the loss
            # also lets save aside the reference to the last object
            # whose output is the models output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Update loss objext with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is softmax and loss functioins is categorical
        # cross-entropy create an object of combined activation and loss
        # function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(
            self.loss, Loss_CategoricalCrossentropy
        ):
            # Create an object of combined activation and loss function
            self.softmax_classifier_output = (
                Activation_Softmax_Loss_CategoricalCrossentropy()
            )

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

    # Train the model
    def train(
        self,
        X,
        y,
        *,
        epochs=1,
        batch_size=None,
        print_every=1,
        validation_data=None,
    ):

        # initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not set
        train_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size

            # Dividing rounds down. If there are some remaining data but not a
            # full batch this wont include it Add 1 to include this not full
            # batch
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main trainning loop
        for epoch in range(1, epochs + 1):

            # Print epoch number
            print(f"epoch: {epoch}")

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # if batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice batch
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]

                # Perform a foward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(
                    output, batch_y, include_regulatization=True
                )
                loss = data_loss + regularization_loss

                # This looks like we have the lack of re drop
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)

                # At this point
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Call backward method on the los
                # this will set dinputs property that the last
                # layer will try to access shortly
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)

                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(
                        f"epoch:{epoch}, "
                        + f"acc: {accuracy:.3f}, "
                        + f"loss: {loss:.3f} ("
                        + f"data_loss: {data_loss:.3f}, "
                        + f"reg_loss: {regularization_loss:.3f}), "
                        + f"lr:{self.optimizer.current_learning_rate}"
                    )

                # Get and print epoch loss and accuracy
            (
                epoch_data_loss,
                epoch_regularization_loss,
            ) = self.loss.calculate_accumulated(include_regulatization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(
                f"training, "
                + f"acc: {epoch_accuracy:.3f}, "
                + f"loss: {epoch_loss:.3f} ("
                + f"data_loss: {epoch_data_loss:.3f}, "
                + f"reg_loss: {epoch_regularization_loss:.3f}), "
                + f"lr: {self.optimizer.current_learning_rate}"
            )

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    # Train the model
    def forward(self, x, training):

        # Call forward method on the input layer
        # this will set the ouput property that
        # this first layer in "prev" objext is expecting
        self.input_layer.forward(x, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "Layer" is now the last object from the list
        # return is output
        # return layer.output
        # phrint("This is X after getting passed through forward")
        return layer.output

    # Performs backward pass
    def backward(self, output, y):

        # If softmax classfier
        if self.softmax_classifier_output is not None:
            # First call backward method on the conbined activation/loss.
            # This will set the dinputs correctly
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer wich is
            # softmax activation/loss object, lets set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through all the objects but last in
            # reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the tools
        # This will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # call backward method going through all the objects
        # In reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):

        # Default value if batch size is not being set
        validation_steps = 1

        # calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            # Dividing round down. If there are some remaining data, but not
            # the full batch this wont include it.
            # Add 1 to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            # If batch size is not set - train using one step and full
            # data base
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # otherwise slice a batch
            else:
                batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)

            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print summary
        print(
            f"validation, "
            + f"acc: {validation_accuracy:.3f}, "
            + f"loss: {validation_loss:.3f}"
        )

    def get_parameters(self):

        # Create a list of parameters
        parameters = []

        # Iterable trainable layers and get their parameteres
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):

        # Iterate over the paramters and layers
        # and update each layers with each set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        # Open a file in the binary-write mode
        # and save parameters to it
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load_parameters(self, path):
        # Open the file where the parameters are
        # witht he weights and biases
        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        # Make a deep copy of the current model
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)

        # For each layer remove inputs, outputs, and dinputs properties
        for layer in model.layers:
            for property in ["inputs", "outputs", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)

        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):

        # Open file in the binary-read mode, load a model
        with open(path, "rb") as f:
            model = pickle.load(f)

        # Return Model
        return model

    def predict(self, x, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # Deviding rounds down. If there are some remaining
            # data, but not a full batch, this wont include it
            # add "1" to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):
            # if batch size is not set
            # train using one step and full data set
            if batch_size is None:
                batch_X = x

            # Otherwise slice a batch
            else:
                batch_X = X[step * batch_size : (step + 1) * batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)


# Common accuracy class
class Accuracy:
    # Calculates an accuracy given predictions and grount thruths values
    def calculate(self, predictions, y):

        # Get comparison results
        comparison = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparison)

        self.accumulated_count = 0
        self.accumulated_sum = 0

        # Add accumulated sum of matching values and count
        self.accumulated_sum += np.sum(comparison)
        self.accumulated_count += len(comparison)
        # Return accuracy
        return accuracy

    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization loss
        return accuracy

    def new_pass(self):
        self.accuracy_sum = 0
        self.accuracy_count = 0


# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precisions value
    # based on passed in graound thruth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode
        self.binary = binary

    def init(self, y):
        pass

    # compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y


def load_mnist_dataset(dataset, path):

    # scan all the directories and create a list of lables
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each lable folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the imgage
            image = cv2.imread(
                os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
            )
            # And append it and a label to the list
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype("uint8")


fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets seperatley
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)

    # And return all data
    return X, y, X_test, y_test


# Create dataset
# X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

# Shuffle the trainning dataset
# keys = np.array(range(X.shape[0]))
# np.random.shuffle(keys)
# X = X[keys]
# y = y[keys]

# Scale and reshape smaples
# X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
#
# model = Model.load("fashion_mnist.model")
#
# confidences = model.predict(X_test[:3])
# print(confidences)
# # model.evaluate(X_test, y_test)
#
# predictions = model.output_layer_activation.predictions(confidences)
#
# for prediction in predictions:
#     print(fashion_mnist_labels[prediction])

image_data = cv2.imread("example_image_pants.png", cv2.IMREAD_GRAYSCALE)

image_data = cv2.resize(image_data, (28, 28))

plt.imshow(image_data, cmap="gray")
plt.show()

image_data = (image_data.reshape(-1, 1).astype(np.float32) - 127.5) / 127.5

model = Model.load("fashion_mnist.model")

confidences = model.predict(image_data)

predictions = model.output_layer_activation.predictions(confidences)

predictions = fashion_mnist_labels[predictions[0]]

print(predictions)
