#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class CategoricalAccuracy(object):
    def __init__(self, name='categorical_accuracy', dtype=None):
        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._dtype = dtype

    def __call__(self, *args, **kwargs):
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        with self._name_scope:
            if sample_weight is None:
                sample_weight = tf.constant(1, dtype=tf.float32)
            y_pred = tf.argmax(y_pred, axis=-1, name='argmax')
            y_true = tf.argmax(y_true, axis=-1)

            correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
            correct = tf.multiply(correct, sample_weight, name='add_weight')
            accuracy = tf.reduce_mean(correct, name='accuracy')
        return accuracy


class SparseCategoricalAccuracy(object):
    def __init__(self, name='categorical_accuracy', dtype=None):
        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._dtype = dtype

    def __call__(self, *args, **kwargs):
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        with self._name_scope:
            if sample_weight is None:
                sample_weight = tf.constant(1, dtype=tf.float32)
            y_pred = tf.cast(tf.argmax(y_pred, axis=-1, name='argmax'), dtype=tf.int32)
            y_true = tf.cast(y_true, dtype=tf.int32)
            correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
            correct = tf.multiply(correct, sample_weight, name='add_weight')
            accuracy = tf.reduce_mean(correct, name='accuracy')
        return accuracy


if __name__ == '__main__':
    pass
