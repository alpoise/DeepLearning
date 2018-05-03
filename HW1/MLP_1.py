#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:04:17 2018

@author: Alpoise
"""

import os
import tensorflow as tf

os.chdir('/Users/Alpoise/Desktop/DL_course')
# LOAD MINIST_DATA
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# BEGINNER'S MINIST 
sess = tf.InteractiveSession()
# Computation map
# variables and connections
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
#y = tf.nn.sigmoid(tf.matmul(x,W) + b)
#y = (tf.nn.tanh(tf.matmul(x,W) + b) + 1) / 2
#y = (tf.nn.softsign(tf.matmul(x,W) + b) + 1) / 2


y_ = tf.placeholder("float", [None,10])
# Define the loss function and training algorithm
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initialization
init = tf.global_variables_initializer()
sess.run(init)
# Train
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()


# MLP with 2 layers; diffferent initialization matters!
sess = tf.InteractiveSession()
x_image = tf.placeholder(tf.float32, [None, 784])
#W_1 = tf.Variable(tf.zeros([784, 100]))
W_1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
#b_1 = tf.Variable(tf.zeros([100]))
b_1 = tf.Variable(tf.constant(0.1, shape=[300]))

h_2 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
#W_2 = tf.Variable(tf.zeros([100, 10]))
W_2 = tf.Variable(tf.truncated_normal([300,10], stddev=0.1))
#b_2 = tf.Variable(tf.zeros([10]))
b_2 = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(h_2,W_2) + b_2)
y_ = tf.placeholder("float", [None,10])
# Define the loss function and training algorithm
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initialization
init = tf.global_variables_initializer()
sess.run(init)
# Train
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()


# MLP with 3 layers
sess = tf.InteractiveSession()
#layer1
x_image = tf.placeholder(tf.float32, [None, 784])
W_1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[100]))
#Layer2
h_2 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
W_2 = tf.Variable(tf.truncated_normal([100,50], stddev=0.5))
b_2 = tf.Variable(tf.constant(0.7, shape=[50]))
#Layer3
h_3 = tf.nn.relu(tf.matmul(h_2,W_2) + b_2)
W_3 = tf.Variable(tf.truncated_normal([50,10], stddev=1))
b_3 = tf.Variable(tf.constant(0.5, shape=[10]))

y = tf.nn.softmax(tf.matmul(h_3,W_3) + b_3)
y_ = tf.placeholder("float", [None,10])
# Define the loss function and training algorithm
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initialization
init = tf.global_variables_initializer()
sess.run(init)
# Train
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()


# Optimization Method and learning rate
sess = tf.InteractiveSession()
x_image = tf.placeholder(tf.float32, [None, 784])
W_1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[300]))

h_2 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
W_2 = tf.Variable(tf.truncated_normal([300,10], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(h_2,W_2) + b_2)
y_ = tf.placeholder("float", [None,10])
# Define the loss function and training algorithm
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#train_step = tf.train.AdagradOptimizer(0.005).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
#train_step = tf.train.AdadeltaOptimizer(0.005).minimize(cross_entropy)
train_step = tf.train.MomentumOptimizer (0.05,2).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initialization
init = tf.global_variables_initializer()
sess.run(init)
# Train
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()


# AutoEncoder
sess = tf.InteractiveSession()
x_image = tf.placeholder(tf.float32, [None, 784])
W_1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[300]))

h_2 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
W_2 = tf.Variable(tf.truncated_normal([300,784], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[784]))

y = tf.nn.sigmoid(tf.matmul(h_2,W_2) + b_2)
y_ = tf.placeholder(tf.float32, [None, 784])

W_3 = tf.Variable(tf.truncated_normal([300,10], stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_3 = tf.nn.softmax(tf.matmul(h_2,W_3) + b_3)
y_out = tf.placeholder("float", [None,10])

# Define the loss function
cross_entropy = -tf.reduce_sum(y_out*tf.log(y_3)) #whole MLP
cost = tf.reduce_mean(tf.pow(y_ - y, 2)) #for autoencoder
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)#autoencoder train
real_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initialization
init = tf.global_variables_initializer()
sess.run(init)
# pre-Train
#print(b_1[1].eval())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_xs})
#print(b_1[1].eval())
# Train
for j in range(1000):
    #print(b_1[1].eval())
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(real_train_step, feed_dict={x: batch_xs, y_out:batch_ys})

# Test
correct_prediction = tf.equal(tf.argmax(y_3,1), tf.argmax(y_out,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_out: mnist.test.labels}))

sess.close()




 






















