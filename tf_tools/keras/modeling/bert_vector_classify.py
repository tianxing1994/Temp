#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from tf_tools.keras import Sequential, Input, Model
from tf_tools.keras.layers.dense import Dense


class BertVectorClassify(object):
    """此处, 我们的类别向量训练得出, 而非句子编码"""
    def __init__(self, clusters_init_value, units_list, clusters_initializer=None, dtype=None, name='bert_vector_classify'):
        self._clusters_init_value = clusters_init_value
        self._units_list = units_list
        self._clusters_initializer = clusters_initializer or tf.glorot_uniform_initializer()
        self._dtype = dtype or tf.float32
        self._name = name
        self._name_scope = tf.name_scope(name=name)

        self._clusters = None
        self._feed_forward = None

    def call(self, x):
        x = self._feed_forward(x)
        c = self._feed_forward(self._clusters)

        x = tf.matmul(x, c, transpose_b=True)
        x = tf.nn.softmax(x)
        return x

    def build(self):
        with self._name_scope:
            self._clusters = tf.constant(self._clusters_init_value, dtype=self._dtype)

            layers = list()
            for i, units in enumerate(self._units_list):
                layers.append(
                    Dense(
                        units=units,
                        activation=tf.nn.relu
                    )
                )
            self._feed_forward = Sequential(layers=layers, name=self._name)

            x = Input(dtype=tf.float32, shape=(None, 768), name='querys')
            y_pred = self.call(x)
            model = Model(x=x, y_pred=y_pred)
        return model


if __name__ == '__main__':
    pass
