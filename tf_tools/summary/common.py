#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim


def model_summary():
    print('\n')
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def model_summary_v2():
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in variables:
        print(v)


if __name__ == '__main__':
    pass
