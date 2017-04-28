# coding=utf-8

import tensorflow as tf
import numpy as np

real_w = [-3, 2]
real_b = 2
x_data = np.float32(np.random.rand(2, 100))
bias = np.random.standard_normal(100)/10
y_data = np.dot(real_w, x_data) + real_b + bias

graph = tf.Graph()
with graph.as_default():
    b = tf.Variable(tf.zeros([1]))      # init param
    w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))       # init param
    y = tf.matmul(w, x_data) + b    # linear model

    # minimize mean loss function
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()     # init all variable


# run graph
with tf.Session(graph=graph) as sess:
    sess.run(init)      # init
    wb_old = np.array(list(sess.run(w)[0]) + list(sess.run(b)))
    # iter train param
    for step in xrange(400):
        sess.run(train)
        wb_new = np.array(list(sess.run(w)[0]) + list(sess.run(b)))
        error = wb_new - wb_old
        wb_old = wb_new
        # if min(np.fabs(error)) < 0.000001:
        #     print '-'*20
        #     print sess.run(w)[0], sess.run(b)
        #     print wb_new
        #     print error
        #     break
        if step % 50 == 0:
            print wb_new
    print wb_new
    print error
