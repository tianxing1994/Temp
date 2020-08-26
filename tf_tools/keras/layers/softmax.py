#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Softmax(object):
    def __init__(self, axis=-1, **kwargs):
        self._axis = axis
        self._name_scope = tf.name_scope(kwargs.get('name', 'softmax'))

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs
            x = tf.nn.softmax(x, axis=self._axis)
        return x


if __name__ == '__main__':
    pass
