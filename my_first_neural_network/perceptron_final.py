import numpy as np 

# This is the sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# This is the derivative of the sigmoid funtion. 
# It returns the slope of the given value. The smaller 
# the value the stipper the slope

def sigmoid_derivative(x):
	return x * (1 - x)

# input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output dataset
training_outputs = np.array([[0,1,0,1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights')
print(synaptic_weights)

for iteration in range(100):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustment = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustment)

print('synaptic weights after trainning')
print(synaptic_weights) 

print('after training outputs')
print(outputs)
