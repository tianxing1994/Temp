#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


def demo1():
    x = np.array(
        [[1, 2, 3], [4, 2, 3], [4, 3, 2], [2, 3, 1], [2, 1, 3], [4, 3, 4]],
        dtype=np.float
    )
    y = np.array(
        [2, 1, 0, 2, 0, 1]
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 3), name='inputs')
    targets = tf.placeholder(dtype=tf.int32, shape=(None,), name='targets')
    outputs = tf.nn.in_top_k(predictions=inputs, targets=targets, k=2)
    ret = session_run(outputs=outputs, feed_dict={inputs: x, targets: y})
    print(ret)
    return


def demo2():
    x = np.array(
        [[1, 2, 3],
         [4, 2, 3],
         [4, 3, 2],
         [2, 3, 1],
         [2, 1, 3],
         [4, 3, 4]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 3), name='inputs')

    outputs = tf.nn.top_k(input=inputs, k=2, sorted=False)
    values, indices = session_run(outputs=outputs, feed_dict={inputs: x})
    print(values)
    print(indices)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
