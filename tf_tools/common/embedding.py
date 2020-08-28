#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class EmbeddingLookup(object):
    def __init__(self, input_dim, output_dim, one_hot=False,
                 name='embedding_lookup', reuse=tf.AUTO_REUSE):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.one_hot = one_hot
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

        self.embeddings = None

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                ids = inputs
                self.embeddings = tf.get_variable(
                    name='embeddings',
                    shape=(self.input_dim, self.output_dim),
                    initializer=tf.truncated_normal_initializer(stddev=0.02)
                )
                # embeddings = tf.gather(params=self.embedding_t, indices=ids)
                embeddings = tf.nn.embedding_lookup(params=self.embeddings, ids=ids)
        return embeddings


def demo1():
    x = np.array(
        [[1, 2, 1, 3, 1],
         [2, 3, 1, 2, 3]],
        dtype=np.int
    )

    inputs = tf.placeholder(dtype=tf.int32, shape=(None, 5), name='inputs')

    embedding_lookup = EmbeddingLookup(input_dim=10, output_dim=3)
    outputs = embedding_lookup(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
