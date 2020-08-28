#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class LayerNormalization(object):
    _epsilon = 1e-7

    def __init__(self, name='layer_normalization', reuse=tf.AUTO_REUSE):
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                epsilon = tf.square(self._epsilon)
                mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
                variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
                std = tf.sqrt(variance + epsilon)
                outputs = (inputs - mean) / std
                gamma = tf.get_variable(name='gamma', shape=inputs.shape[-1], initializer=tf.ones_initializer)
                beta = tf.get_variable(name='beta', shape=inputs.shape[-1], initializer=tf.zeros_initializer)
                outputs *= gamma
                outputs += beta
        return outputs


def demo1():
    x = np.array(
        [[[1, 2, 3],
          [2, 3, 4],
          [3, 4, 5],
          [4, 5, 6]]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4, 3), name='inputs')
    ln = LayerNormalization()
    outputs = ln(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
