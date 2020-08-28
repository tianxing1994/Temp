#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Dense(object):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer=None,
                 bias_initializer=None, name='dense', reuse=tf.AUTO_REUSE):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                kernel = tf.get_variable(
                    name='kernel',
                    shape=(inputs.shape[-1], self.units),
                    dtype=tf.float32,
                    initializer=self.kernel_initializer
                )
                o = tf.reshape(inputs, shape=(-1, tf.shape(inputs)[-1]))
                o = tf.matmul(o, kernel)
                new_shape = tf.concat(
                    [tf.shape(inputs)[:-1],
                     tf.constant(value=self.units, shape=(1,))],
                    axis=0
                )
                o = tf.reshape(tensor=o, shape=new_shape)
                if self.use_bias:
                    bias = tf.get_variable(
                        name='bias',
                        shape=(self.units,),
                        dtype=tf.float32,
                        initializer=self.bias_initializer
                    )
                    o = tf.add(o, bias)
                if self.activation is not None:
                    o = self.activation(o)
        return o


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
        use_bias=False,
    )

    outputs = dense(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
