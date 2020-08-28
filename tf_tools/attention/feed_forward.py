#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class FeedForward(object):
    """FeedForward class (Position-wise Feed-Forward Networks)"""

    def __init__(self, hidden_size=2048, model_dim=512, dropout=0.1,
                 name='feed_forward', reuse=tf.AUTO_REUSE):
        self.model_dim = model_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                result = self.dense_relu_dense(inputs)
                # result = self.conv_relu_conv(inputs)
        return result

    def dense_relu_dense(self, inputs):
        o = tf.layers.dense(inputs, self.hidden_size, activation=tf.nn.relu, name='dense_1')
        o = tf.layers.dense(o, self.model_dim, name='dense_2')
        o = tf.nn.dropout(o, 1.0 - self.dropout)
        return o

    def conv_relu_conv(self, inputs):
        o = tf.layers.conv1d(inputs, filters=self.hidden_size, kernel_size=1, padding='SAME', name='conv1d_1')
        o = tf.nn.relu(o)
        o = tf.layers.conv1d(o, filters=self.model_dim, kernel_size=1, padding='SAME', name='conv1d_2')
        o = tf.nn.dropout(o, 1.0 - self.dropout)
        return o


def demo1():
    x = np.array(
        [[[1, 2, 3, 1],
          [3, 2, 1, 3],
          [3, 2, 1, 3],
          [3, 2, 1, 3],
          [4, 3, 2, 4],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 7, 4), name='inputs')
    feedforward = FeedForward(hidden_size=8, model_dim=4)
    outputs = feedforward(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
