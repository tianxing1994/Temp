#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class PositionalEmbedding(object):
    m_add = 0
    m_concat = 1

    t_random = 0
    t_sin_cos = 1

    def __init__(self, input_dim, output_dim, embedding_type=0, merge_mode=0,
                 name='positional_embedding', reuse=tf.AUTO_REUSE):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_type = embedding_type
        self.merge_mode = merge_mode
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(
            name_or_scope=name,
            reuse=reuse
        )
        self.reuse = reuse

    @staticmethod
    def _random_positional_embedding(seq_len, emb_size, initializer=tf.truncated_normal_initializer):
        embeddings = tf.get_variable(
            name='embeddings',
            shape=(seq_len, emb_size),
            initializer=initializer
        )
        return embeddings

    @staticmethod
    def _sin_cos_positional_embedding(seq_len, emb_size):
        encoded_vec = np.array([
            pos / np.power(10000, 2 * i / emb_size)
            for pos in range(seq_len)
            for i in range(emb_size)
        ])
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        embeddings = tf.convert_to_tensor(
            encoded_vec.reshape([seq_len, emb_size]),
            dtype=tf.float32
        )
        return embeddings

    def get_embeddings(self, seq_len, emb_size):
        if self.embedding_type == self.t_random:
            embeddings = self._random_positional_embedding(seq_len, emb_size)
        elif self.embedding_type == self.t_sin_cos:
            embeddings = self._sin_cos_positional_embedding(seq_len, emb_size)
        else:
            embeddings = self._random_positional_embedding(seq_len, emb_size)
        return embeddings

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                f = lambda x: int(x) if x._value else -1
                s = tuple(map(f, inputs.shape))
                ndim = len(s)
                seq_len = tf.shape(inputs)[-2]

                embeddings = self.get_embeddings(self.input_dim, self.output_dim)
                embeddings = embeddings[:seq_len]

                for _ in range(ndim - 2):
                    embeddings = tf.expand_dims(embeddings, axis=0)
                if self.merge_mode == self.m_add:
                    outputs = inputs + embeddings
                else:
                    multiples = tf.concat(
                        [tf.shape(inputs)[:-2], tf.constant([1, 1], dtype=tf.int32)],
                        axis=0
                    )
                    embeddings = tf.tile(input=embeddings, multiples=multiples)
                    outputs = tf.concat([inputs, embeddings], axis=-1)
        return outputs


def demo2():
    x = np.array(
        [[[1, 2, 3],
          [2, 3, 4],
          [3, 4, 5],
          [4, 5, 6]],

         [[6, 5, 4],
          [5, 4, 3],
          [4, 3, 2],
          [3, 2, 1]]],
        dtype=np.float)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4, 3), name='inputs')

    # pe = PositionalEmbedding(input_dim=4, output_dim=3, embedding_type=PositionalEmbedding.t_random, merge_mode=1)
    # pe = PositionalEmbedding(input_dim=4, output_dim=3, embedding_type=PositionalEmbedding.t_sin_cos, merge_mode=1)
    pe = PositionalEmbedding(input_dim=4, output_dim=3, embedding_type=PositionalEmbedding.t_sin_cos, merge_mode=0)

    outputs = pe(inputs=inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
