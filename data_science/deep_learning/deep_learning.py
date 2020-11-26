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


def neural_network_model(data):

    hidden_layer1 = {
        "weights": tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        "biases": tf.Variable(tf.random_normal(n_nodes_hl1)),
    }

    hidden_layer2 = {
        "weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        "biases": tf.Variable(tf.random_normal(n_nodes_hl2)),
    }

    hidden_layer3 = {
        "weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        "biases": tf.Variable(tf.random_normal(n_nodes_hl3)),
    }

    output_layer = {
        "weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        "biases": tf.Variable(tf.random_normal(n_classes)),
    }

    # (input_data * weights ) + biases
    l1 = tf.add(
        tf.matmul(data, hidden_layer1["weights"]), hidden_layer1["biases"]
    )
    l1 = tf.nn.relu(l1)

    l2 = tf.add(
        tf.matmul(l1, hidden_layer2["weights"]), hidden_layer2["biases"]
    )
    l3 = tf.nn.relu(l2)

    l3 = tf.add(
        tf.matmul(l2, hidden_layer3["weights"]), hidden_layer3["biases"]
    )
    l1 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer["weights"] + output_layer["biases"])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(prediction, y)
    )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in hm_epochs:
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_sizes)):
                x, y = mnist.train.next_batch(batch_sizes)
                _, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})
                epoch_loss += c
            print("epoch", epoch)
