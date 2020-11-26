"""
input > weight > hidden layer 1 (activation function) >  weights > hidden layer 2
(activation function) > output layer

compare the output to the intended output > cost function (cross entropy)
optimization function (optimizer) > minimize that cost (adamoptimizer, sgd, adagrad)

backpropacation

feer forward + backprop = epoch
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_sizes = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network_model():
    # (input_data * weights ) + biases
    hidden_layer1 = {
        "weights": tf.Variable(
            tf.random_normal_initializer([784, n_nodes_hl1])
        ),
        "biases": tf.Variable(tf.random_normal_initializer(n_nodes_hl1)),
    }

    hidden_layer1 = {
        "weights": tf.Variable(
            tf.random_normal_initializer([784, n_nodes_hl1])
        ),
        "biases": tf.Variable(tf.random_normal_initializer(n_nodes_hl1)),
    }

    hidden_layer1 = {
        "weights": tf.Variable(
            tf.random_normal_initializer([784, n_nodes_hl1])
        ),
        "biases": tf.Variable(tf.random_normal_initializer(n_nodes_hl1)),
    }

    hidden_layer1 = {
        "weights": tf.Variable(
            tf.random_normal_initializer([784, n_nodes_hl1])
        ),
        "biases": tf.Variable(tf.random_normal_initializer(n_nodes_hl1)),
    }
