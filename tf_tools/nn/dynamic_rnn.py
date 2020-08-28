#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tf_tools.debug_tools.common import session_run


def dynamic_rnn(cell, inputs, initial_state=None, dtype=tf.float32):
    if initial_state is None:
        batch_size = tf.shape(inputs)[0]
        initial_state = cell.zero_state(batch_size=batch_size, dtype=dtype)

    _, s_len, _ = inputs.shape
    input_list = tf.split(inputs, num_or_size_splits=s_len, axis=1)

    y_list = list()
    for i, input in enumerate(input_list):
        input = tf.squeeze(input, axis=1)
        if i == 0:
            y_, states = cell(input, initial_state)
            y_ = tf.expand_dims(y_, axis=-2)
            y_list.append(y_)
        else:
            y_, states = cell(input, states)
            y_ = tf.expand_dims(y_, axis=-2)
            y_list.append(y_)
    y = tf.concat(y_list, axis=-2)
    return y, states


def demo1():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')
    multiples = (tf.shape(inputs)[0], 1)
    initial_state = tf.get_variable(name='initial_state', shape=(1, 3), dtype=tf.float32)
    initial_state = tf.tile(input=initial_state, multiples=multiples)

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)

    outputs, state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=inputs,
        initial_state=initial_state,
        dtype=tf.float32
    )
    y, h = session_run([outputs, state], feed_dict={inputs: x})
    print(y)
    print(h)
    return


def demo2():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')
    multiples = (tf.shape(inputs)[0], 1)
    initial_state = tf.get_variable(name='initial_state', shape=(1, 3), dtype=tf.float32)
    initial_state = tf.tile(input=initial_state, multiples=multiples)

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)

    outputs, state = dynamic_rnn(
        cell=rnn_cell,
        inputs=inputs,
        initial_state=initial_state,
        dtype=tf.float32
    )
    y, h = session_run([outputs, state], feed_dict={inputs: x})
    print(y)
    print(h)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
