#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Matmul(object):
    def __init__(self, activation=None, initializer=None, name='bert_matmul', reuse=tf.AUTO_REUSE):
        self.activation = activation
        self.initializer = initializer
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, a, b, transpose_a=False, transpose_b=False, use_bias=True):
        with self.variable_scope:
            with self.name_scope:
                if transpose_a:
                    a = tf.transpose(a)
                if transpose_b:
                    b = tf.transpose(b)
                f = lambda x: int(x) if x._value else -1
                sb = tuple(map(f, b.shape))

                o = tf.reshape(a, shape=(-1, tf.shape(a)[-1]))
                o = tf.matmul(a=o, b=b)
                new_shape = tf.concat(
                    [tf.shape(a)[:-1],
                     tf.constant(sb[-1], shape=(1,))],
                    axis=0
                )
                o = tf.reshape(tensor=o, shape=new_shape)
                if use_bias:
                    units = sb[-1]
                    bias = tf.get_variable(
                        name='bias',
                        shape=(units,),
                        dtype=tf.float32,
                        initializer=self.initializer
                    )
                    o = tf.add(o, bias)
                o = self.activation(o)
        return o


def demo1():
    a = np.array(
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

    b = np.array(
        [[1, 1],
         [1, 1],
         [1, 1],
         [1, 1]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')
    kernel = tf.placeholder(dtype=tf.float32, shape=(4, 2), name='kernel')

    bert_matmul = Matmul(
        activation=tf.nn.softmax,
    )

    outputs = bert_matmul(inputs, kernel)

    ret = session_run(outputs, feed_dict={inputs: a, kernel: b})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
