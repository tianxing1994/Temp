#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .model import Model


class Sequential(Model):
    def __init__(self, layers, name='sequential'):
        super(Sequential, self).__init__()
        self.layers = layers
        self.name_scope = tf.name_scope(name=name)

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        with self.name_scope:
            x = inputs
            for layer in self.layers:
                x = layer(x)
        return x


if __name__ == '__main__':
    pass
