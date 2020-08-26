#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Concatenate(object):
    def __init__(self, axis=-1, **kwargs):
        self._axis = axis
        self._name_scope = tf.name_scope(name=kwargs.get('name', 'concatenate'))

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs
            x = tf.concat(x, axis=self._axis)
        return x


if __name__ == '__main__':
    pass
