# back propagation on a single neuron without the use of any library


x = [1, -2, 3]
w = [-3, -1, 2]
b = 1

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b


# relu function
def relu(x):
    if x > 0:
        return x
    else:
        return 0


# derivative of the relu function
def reluPrime(z):
    if z > 0:
        return 1
    else:
        return 0


print("learning this one liner")
z = -1
print((1 if z > 0 else 0))

# def sumPrime(num1, num2, num3):


neuron_output = relu(z)
print(neuron_output)


next_layer_value = 1
# backpropagation

# we need to find the gradiant verctor for these
# x_backpropagation = derWrespecToX(relu(sum(mul(weights and inputs) and bias)))
# w_backpropagation = derWrespectToW(relu(sum(mul(weights and inputs) and bias)))
# b_backpropagation = derWrespectToB(relu(sum(mul(weights and inputs) and bias)))

# Multiply activations function derivative with the derivative value recievd
# from the next layer which is 1
deri_relu = next_layer_value * reluPrime(neuron_output)
print(deri_relu)

# the partial derivative of a sum is always 1. For example y = x + y + z,
# y'Wx = 1. Bc the constants become 0 and the derivative of x equals 1
deri_sum = 1
deri_sum = deri_relu * deri_sum
