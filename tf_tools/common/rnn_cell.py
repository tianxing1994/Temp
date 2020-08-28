#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

from tf_tools.debug_tools.common import session_run


class BasicRNNCell(object):
    def __init__(self, num_units, activation=None, name='basic_rnn_cell', dtype=None):
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self.name_scope = tf.name_scope(name=f'{name}_op_node')
        self.variable_scope = tf.variable_scope(name_or_scope=f'{name}')
        self._dtype = dtype or tf.float32

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype=tf.float32):
        multiples = (batch_size, 1)
        state = tf.Variable(tf.zeros(shape=(1, self._num_units)), dtype=dtype, name='initial_state')
        state = tf.tile(input=state, multiples=multiples)
        return state

    def __call__(self, inputs, state):
        with self.variable_scope:
            with self.name_scope:
                x = tf.concat([inputs, state], axis=-1)
                kernel = tf.get_variable(
                    shape=(x.shape[-1]._value, self._num_units),
                    initializer=tf.random_normal_initializer,
                    dtype=self._dtype,
                    name='kernel'
                )
                bias = tf.get_variable(
                    shape=(self._num_units,),
                    initializer=tf.random_normal_initializer,
                    dtype=self._dtype,
                    name='bias'
                )
                o = tf.matmul(x, kernel)
                o = tf.add(o, bias, name='BiasAdd')
                o = self._activation(o, name=self._activation.__name__)
        return o, o


def demo1():
    x = np.array(
        [[1, 2, 3, 4],
         [4, 3, 2, 1]],
        dtype=np.float
    )

    rnn_cell = BasicRNNCell(num_units=3)
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='inputs')
    batch_size = tf.shape(inputs)[0]
    state = rnn_cell.zero_state(batch_size)
    outputs = rnn_cell(inputs, state)

    y, h = session_run(outputs, feed_dict={inputs: x})
    print(y)
    print(h)
    return


def demo2():
    x = np.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='inputs')
    multiples = (tf.shape(inputs)[0], 1)
    state = tf.get_variable(name='initial_state', shape=(1, 3), dtype=tf.float32)
    state = tf.tile(input=state, multiples=multiples)

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)

    outputs = rnn_cell(inputs, state)
    y, h = session_run(outputs, feed_dict={inputs: x})
    print(y)
    print(h)
    return


def demo3():
    x = np.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='inputs')

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)
    state = rnn_cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
    outputs = rnn_cell(inputs, state)
    y, h = session_run(outputs, feed_dict={inputs: x})
    print(y)
    print(h)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
