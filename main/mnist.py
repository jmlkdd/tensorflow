# coding=utf-8

import tensorflow as tf
from utils.load_minst_data import load_minst


data_sets = load_minst(path='../datasets', one_hot=True)


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, name='x')
    y_ = tf.placeholder(tf.float32, name='y_')
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()


with tf.Session(graph=graph) as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = data_sets.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run(accuracy, feed_dict={x: data_sets.test.images, y_: data_sets.test.labels})
