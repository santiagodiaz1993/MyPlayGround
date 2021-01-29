# import matplotlib.pylot as plt
import numpy as np


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


input = [-1, -0.08, -0.6, -0.4, -0.2, 0, 0.02, 0.4, 0.6, 0.8, 1]
output_array = []
for element in input:
    print("element is" + str(element))
    output = relu((element * -1) + 0.5)
    output2 = relu((output * -2) + 1)
    output_array.append(output2)

plt.plot(output_array)
plt.show()
