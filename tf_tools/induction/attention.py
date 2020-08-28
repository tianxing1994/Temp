#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Attention(object):
    """
    注意力机制.
    对句子中的词向量加权求和, 以用一个向量表示一个句子.
    此实现通过与权重矩阵作向量内积来计算词权重, 没有考虑位置信息 (位置信息可在 embedding 时被包含到特征中).
    """
    def __init__(self, feature_size, attention_size, name='attention', reuse=tf.AUTO_REUSE):
        self._feature_size = feature_size
        self._attention_size = attention_size
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs):
        """
        :param inputs: tensor, shape=(batch_size, seq_len, hidden_size).
        :return:
        """
        with self.variable_scope:
            with self.name_scope:
                w_1 = tf.get_variable(
                    name='w_1',
                    shape=(self._feature_size, self._attention_size),
                    dtype=tf.float32
                )

                w_2 = tf.get_variable(
                    name='w_2',
                    shape=(self._attention_size,),
                    dtype=tf.float32
                )

                x = inputs
                seq_len = tf.shape(x)[-2]
                x = tf.reshape(x, shape=(-1, self._feature_size))
                m = tf.tanh(tf.matmul(x, w_1))

                # shape=(batch_size, seq_len)
                weights = tf.reshape(
                    tf.matmul(m, tf.reshape(w_2, shape=(-1, 1))),
                    shape=(-1, seq_len)
                )

                alpha = tf.nn.softmax(weights, axis=-1)
                outputs = tf.reduce_sum(inputs * tf.reshape(alpha, shape=(-1, seq_len, 1)), axis=1)
        return outputs


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
    attention = Attention(
        feature_size=4,
        attention_size=12
    )
    outputs = attention(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
