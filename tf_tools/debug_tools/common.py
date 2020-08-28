#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_tools.config import SummaryConfig


def model_summary():
    print('\n')
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    return


def model_summary_v2():
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in variables:
        print(v)
    return


def session_run(outputs, feed_dict=None):
    model_summary()

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        with tf.summary.FileWriter(SummaryConfig.outputs_path, sess.graph):
            sess.run(tf.global_variables_initializer())
            ret = sess.run(outputs, feed_dict=feed_dict)
    return ret


if __name__ == '__main__':
    pass
