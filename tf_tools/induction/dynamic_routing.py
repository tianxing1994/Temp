#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class DynamicRoutingBlock(object):
    def __init__(self, num_classes, num_support, encode_size, name='dynamic_routing_block', reuse=tf.AUTO_REUSE):
        self._num_classes = num_classes
        self._num_support = num_support
        self._encode_size = encode_size

        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs, init_b):
        with self.variable_scope:
            with self.name_scope:
                x = inputs
                w_s = tf.get_variable(
                    name='w_s',
                    shape=(self._encode_size, self._encode_size),
                    dtype=tf.float32
                )
                support_trans = tf.reshape(
                    tf.matmul(tf.reshape(x, shape=(-1, self._encode_size)), w_s),
                    shape=(self._num_classes, self._num_support, self._encode_size)
                )
                norm_b = tf.nn.softmax(tf.reshape(init_b, shape=(self._num_classes, self._num_support, 1)), axis=1)
                c_i = tf.reduce_sum(tf.multiply(support_trans, norm_b), axis=1)
                c_squared_norm = tf.reduce_sum(tf.square(c_i), axis=1, keepdims=True)
                scalar_factor = c_squared_norm / (1 + c_squared_norm) / tf.sqrt(c_squared_norm + 1e-9)
                c_squared = scalar_factor * c_i

                c_e_dot = tf.matmul(support_trans, tf.reshape(c_squared, shape=(self._num_classes, self._encode_size, 1)))
                init_b += tf.reshape(c_e_dot, shape=(self._num_classes, self._num_support))
        return c_squared, init_b


class DynamicRouting(object):
    """
    动态路由算法.
    通过迭代, 求解各个类别向量的聚类中心向量.
    """
    def __init__(self, num_classes, num_support, encode_size, n_iters=3, share_block=False, name='dynamic_routing', reuse=tf.AUTO_REUSE):
        self._num_classes = num_classes
        self._num_support = num_support
        self._encode_size = encode_size

        self._n_iters = n_iters
        self._share_block = share_block

        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                init_b = tf.constant(0.0, dtype=tf.float32, shape=(self._num_classes, self._num_support))
                dynamic_routing_block = None
                for i in range(self._n_iters):
                    if self._share_block:
                        if dynamic_routing_block is None:
                            name = 'dynamic_routing_block'
                            dynamic_routing_block = DynamicRoutingBlock(
                                num_classes=self._num_classes,
                                num_support=self._num_support,
                                encode_size=self._encode_size,
                                name=name
                            )
                    else:
                        name = f'dynamic_routing_block_{i+1}'
                        dynamic_routing_block = DynamicRoutingBlock(
                            num_classes=self._num_classes,
                            num_support=self._num_support,
                            encode_size=self._encode_size,
                            name=name
                        )
                    c_squared, init_b_i = dynamic_routing_block(inputs=inputs, init_b=init_b)
                    init_b += init_b_i
        return c_squared


def demo1():
    x = np.array(
        [[[1, 2, 3, 4],
          [4, 3, 2, 1],
          [3, 2, 2, 3],
          [3, 4, 4, 3],
          [1, 3, 2, 1]],

         [[6, 2, 5, 4],
          [4, 3, 2, 1],
          [3, 4, 4, 3],
          [3, 2, 2, 3],
          [1, 3, 2, 1]]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='inputs')
    dynamic_routing = DynamicRouting(
        num_classes=2,
        num_support=5,
        encode_size=4,
        n_iters=10,
        share_block=True
    )
    outputs = dynamic_routing(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
