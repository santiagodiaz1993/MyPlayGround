inputs = [1, 2, 3, 2.5]


weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

biases = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = []
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)


# this is a matric multiplication. This can be scaled with a for loop
output = [
    inputs[0] * weights1[0]
    + inputs[1] * weights1[1]
    + inputs[2] * weights1[2]
    + inputs[3] * weights1[3]
    + bias1,
    inputs[0] * weights2[0]
    + inputs[1] * weights2[1]
    + inputs[2] * weights2[2]
    + inputs[3] * weights2[3]
    + bias2,
    inputs[0] * weights3[0]
    + inputs[1] * weights3[1]
    + inputs[2] * weights3[2]
    + inputs[3] * weights3[3]
    + bias3,
]


print(output)
