import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (sigma ** 2))


x_data = [1, 40, 3, 34, 0]
y_data = [1, 3, 32, 49, 29]
z_data_linear = []
z_data_polynomial = []
z_data_gaussian = []

for i in x_data:
    z_data_linear.append(linear_kernel(i, i))
    z_data_polynomial.append(polynomial_kernel(i, i))
    z_data_gaussian.append(gaussian_kernel(i, i))


plt.plot(x_data, y_data)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    x_data, y_data, z_data_linear, c="r", marker="o",
)
plt.show()

ax.scatter()
