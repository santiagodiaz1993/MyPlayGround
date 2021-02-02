import numpy as np

# Exmaple later outuput
z = np.array([[1, 2, -3, -4], [2, -7, -1, 3], [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# ReLu application to z. Since the derivative of the relu function just returns
# a one when the value is greater than zero, this makes a matrics that when
# multiplyed just keeps values where the 1 are and turns the values into 0
# where the 0 are
drelu = np.copy(z)
drelu[z <= 0] = 0

print(drelu)
