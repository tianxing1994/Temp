#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class CategoricalCrossentropy(object):
    def __init__(self, from_logits=False, label_smoothing=0, name='categorical_crossentropy'):
        self._from_logits = from_logits
        self._label_smoothing = label_smoothing
        self._name = name
        self._name_scope = tf.name_scope(self._name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        with self._name_scope:
            if sample_weight is None:
                sample_weight = tf.constant(1, dtype=tf.float32)
            losses = tf.nn.weighted_cross_entropy_with_logits(
                logits=y_pred,
                labels=y_true,
                pos_weight=sample_weight
            )
            losses = tf.reduce_mean(losses)
        return losses


def demo1():
    logits = tf.constant(value=[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [0, 0, 1]], dtype=tf.float32)

    labels = tf.constant(value=[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [0, 0, 1]], dtype=tf.float32)
    cce = CategoricalCrossentropy()
    outputs = cce(y_true=logits, y_pred=labels)
    ret = session_run(outputs)
    print(ret)
    return


if __name__ == '__main__':
    demo1()
