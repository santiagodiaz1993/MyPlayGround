import numpy as np

softmax_outputs = np.array(
    [[0.07, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
)
class_target = [0, 1, 1]
# print([numpyarray[rows to be extracted], )
print(
    softmax_outputs[[0, 1, 1], [1, 1, 1]]
)  # this will extract values as with "coordinate inputs". First array with X values and seond with Y values
# print(softmax_outputs[[0, 1, 2], class_target])
