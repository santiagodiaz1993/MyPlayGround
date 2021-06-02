# cmap/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nnfs

# np.set_printoptions(linewidth=200)
#
# lables = os.listdir("fashion_mnist_images/train")
# print(lables)
#
# files = os.listdir("fashion_mnist_images/train/7")
# print(files[:10])
# print(len(files))
#
#
# # the second parameter means that we intend to read the imgaes in the same way
# # as they were saved (Grey scaled in this case)
# image_data = cv2.imread(
#     "fashion_mnist_images/train/7/0002.png", cv2.IMREAD_UNCHANGED
# )
#
# print(image_data)

# We can also use matplot lib to visualize the array of values
# And we can also specify the type of image to draw, in this case grey scale
# plt.imshow(image_data, cmap="gray")
# plt.show()


def load_mnist_dataset(dataset, path):

    # scan all the directories and create a list of lables
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each lable folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the imgage
            image = cv2.imread(
                os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
            )
            # And append it and a label to the list
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype("uint8")


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets seperatley
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)

    # And return all data
    return X, y, X_test, y_test


# Neural networks usually work better on a small range. For this reason we will
# scale the pictures
X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(X_test.min(), X.max())


print(X.min(), X.max())

print(X.shape)


# Our neural network only takes 1 dimentional data so this means that we hav
# eto flatten out our images from a matrix into a vector
# X = X.reshape(X.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

keys = np.array(range(X.shape[0]))


nnfs.init()

np.random.shuffle(keys)


print(keys[:10])
print("These are the y valuees")
print(y[0:10])
print(y[6000:6010])

X = X[keys]
y = y[keys]

print(y[:15])

# we can also check indivusual examples as well
# just choose an example and reshape it
plt.imshow((X[8].reshape(28, 28)))
plt.show()

# we cna check the class at the same index. This means that both the labels and
# data is not explicilty mapped. This means the mapping can be easily lost
# which would cause incorrect labeling
print("This is the label")
print(y[8])
