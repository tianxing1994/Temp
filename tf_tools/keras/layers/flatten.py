#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Flatten(object):
    def __init__(self, **kwargs):
        self._name_scope = tf.name_scope(name=kwargs.get('name', 'flatten'))

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs
            if isinstance(x, list):
                temp = list()
                for x_i in x:
                    batch_size = tf.shape(x_i)[0]
                    dim = self.get_dim(x_i.shape)
                    x_i = tf.reshape(x_i, shape=(batch_size, dim))
                    temp.append(x_i)
                x = temp
            else:
                batch_size = tf.shape(x)[0]
                dim = self.get_dim(x.shape)
                x = tf.reshape(x, shape=(batch_size, dim))
        return x

    def get_dim(self, shape):
        d = 1
        for i, dim in enumerate(shape):
            if i == 0:
                continue
            if dim._value is None:
                d = None
                # 如果 d=None 导致报错, 试一下 d=-1.
                break
            d *= dim._value
        return d


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

    inputs = tf.placeholder(dtype=tf.int32, shape=(None, 5, 4), name='inputs')
    flatten = Flatten(input_dim=10, output_dim=4)


    outputs = flatten(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
