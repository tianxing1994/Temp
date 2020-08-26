#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Embedding(object):
    def __init__(self, input_dim, output_dim, embeddings_initializer=None,
                 embeddings_regularizer=None, activity_regularizer=None,
                 embeddings_constraint=None, mask_zero=False, input_length=None, **kwargs):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._embeddings_initializer = embeddings_initializer or tf.random_normal_initializer()

        self._dtype = kwargs.get('dtype', tf.float32)
        self._name_scope = tf.name_scope(name=kwargs.get('name', 'embedding'))

        self._embeddings = None

    def __call__(self, inputs):
        """
        :param inputs: 2D tensor with shape: (batch_size, input_length).
        :return: 3D tensor with shape: (batch_size, input_length, output_dim).
        """
        with self._name_scope:
            ids = tf.cast(inputs, dtype=tf.int32)

            if self._embeddings is None:
                self._embeddings = tf.Variable(
                    self._embeddings_initializer(shape=(self._input_dim, self._output_dim)),
                    dtype=self._dtype,
                    name='embeddings'
                )
            outputs = tf.nn.embedding_lookup(params=self._embeddings, ids=ids)
        return outputs


def demo1():
    x = np.array(
        [[1, 2, 3],
         [6, 5, 4]],
        dtype=np.int
    )

    embedding = Embedding(input_dim=10, output_dim=4)

    inputs = tf.placeholder(dtype=tf.int32, shape=(None, 3), name='inputs')
    outputs = embedding(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
