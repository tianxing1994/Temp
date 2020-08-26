#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def Input(shape=None, batch_size=None, name=None, dtype=None,
          sparse=False, tensor=None, ragged=False, **kwargs):
    outputs = tf.placeholder(dtype=dtype, shape=shape, name=name)
    return outputs


if __name__ == '__main__':
    pass
