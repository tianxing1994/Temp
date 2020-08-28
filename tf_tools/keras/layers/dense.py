#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Dense(object):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer=None, bias_initalizer=None, dtype=tf.float32, name='dense'):
        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        # glorot_uniform_initializer() 调用时好像 shape 不能传张量, 可以用 `tf.truncated_normal_initializer`
        self._kernel_initializer = kernel_initializer or tf.glorot_uniform_initializer()
        self._bias_initalizer = bias_initalizer or tf.zeros_initializer()
        self._dtype = dtype
        self._name_scope = tf.name_scope(name=name)

        self._kernel = None
        self._bias = None

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        with self._name_scope:
            x = inputs
            if self._kernel is None:
                input_dim = x.shape[-1]

                self._kernel = tf.Variable(
                    self._kernel_initializer(shape=(input_dim, self._units)),
                    dtype=self._dtype,
                    name='kernel'
                )
            if self._use_bias:
                if self._bias is None:
                    self._bias = tf.Variable(
                        self._bias_initalizer(shape=(self._units,)),
                        dtype=self._dtype, name='bias'
                    )

            x = tf.matmul(x, self._kernel)
            if self._use_bias:
                x = tf.add(x, self._bias)
            if self._activation is not None:
                x = self._activation(x)
        return x


def demo1():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2]]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')

    dense = Dense(
        units=2,
        use_bias=True,
    )

    outputs = dense(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
