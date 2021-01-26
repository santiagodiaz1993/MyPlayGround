# back propagation on a single neuron without the use of any library

# foward pass
x = [1, -2, 3]
w = [-3, -1, 2]
b = 1

# Multiplication inputs by the weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# adding the weighted inputs and biases
z = xw0 + xw1 + xw2 + b

# Relu activation function
y = max(z, 0)


# back propagations starts here assuming the derivative from the next layers is
# 1
dvalue = 1

# Derivative of the reul function and the chain rule
# derivative value form the next layer times current layer after doing the
# derivative and pluging in the z value
drelu_dz = dvalue * (1 if z > 0 else 0)

# Partial derivatives of the multiplication, the chain rule
# The derivative of all these values are the partial derivatives of
# z = xw0 + xw1 + xw2 + b
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
# All of the them follow a similar partial bc are in the same function
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)
