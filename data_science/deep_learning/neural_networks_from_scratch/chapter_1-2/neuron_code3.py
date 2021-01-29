"""
Dependencies: We need to understand dot product. Dop product does not
equal matric multiplication.

Also matrix addition.
"""

import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, 0.27, 0.17, 0.87],
]

bias = [2, 3, 0.5]


output = np.dot(weights, weights) + bias
print(output)
