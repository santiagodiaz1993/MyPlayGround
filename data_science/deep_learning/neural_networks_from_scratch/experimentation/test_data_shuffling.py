#!/usr/bin/env python
# coding=utf-8
import numpy as np

array = [[1, 2, 3, 4, 1], [3, 5, 6, 7, 8], [1, 3, 4, 6, 4]]
print(array)

numpy_array = np.array(array)
print(numpy_array)
print("This is how to show ranges")
print(numpy_array[:1])


keys = np.array(range(numpy_array.shape[0]))
print("This is the keys")
print(keys)
print("This is after passing keys in into []")

# When we pass in an array into another array with [] then we are poviding the
# index reassignment
np.random.shuffle(keys)
print(numpy_array[keys[:2]])


print("This is what the reshape function does")
print(numpy_array)
print(numpy_array.reshape(5, 3))
