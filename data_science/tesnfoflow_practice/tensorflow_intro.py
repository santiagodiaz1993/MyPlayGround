#!/usr/bin/env python
# coding=utf-8


import tensorflow.compat.v1 as tf

# disabling v3 behaviour
tf.disable_v2_behavior()

# deffining the nodes
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node: ", node3)

# this method allowcates resources for the computation of graphs
sess = tf.Session()

print("session.run(node3) :", sess.run(node3))

# place holder is a variable for assign data to a leter date
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_node = a + b

print(sess.run(add_node, {a: 3, b: 4.5}))
print(sess.run(add_node, {a: 3, b: 4.5}))

add_and_triple = add_node * 3

print(sess.run(add_and_triple, {a: 3, b: 4.5}))


W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
