# coding=utf-8

import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    foo = tf.Variable(3, name='foo')
    bar = tf.Variable(2, name='bar')
    result = foo + bar
    initialize = tf.initialize_all_variables()

    print result

with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    res = sess.run(result)

print res
print tf.__version__
print tf.__path__

foo = tf.placeholder(tf.int32, shape=[1], name='foo')
bar = tf.placeholder(tf.int32, shape=[1], name='bar')

result = foo + bar

with tf.Session() as sess:
    print sess.run(result, {foo: [36], bar: [682]})
