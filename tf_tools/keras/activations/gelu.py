#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


def gelu_erf(inputs):
    outputs = 0.5 * inputs * (1.0 + tf.erf(inputs / np.sqrt(2.0)))
    return outputs


def gelu_tanh(inputs):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (inputs + 0.044715 * pow(inputs, 3)))))
    outputs = inputs * cdf
    return outputs


def demo1():
    x = np.array(
        [[1], [2]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='inputs')

    outputs = gelu_tanh(inputs)
    ret = session_run(outputs=outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
