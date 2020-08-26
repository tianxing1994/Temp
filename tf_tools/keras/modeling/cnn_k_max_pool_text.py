#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.keras import Sequential
from tf_tools.keras.layers.concatenate import Concatenate
from tf_tools.keras.layers.dense import Dense
from tf_tools.keras.layers.embedding import Embedding
from tf_tools.keras.layers.flatten import Flatten
from tf_tools.keras.layers.softmax import Softmax
from tf_tools.keras.layers.k_maxpool import DynamicKMaxPooling

from tf_tools.debug_tools.common import session_run


class CNNKMaxPoolTextModel(object):
    """
    这个实现是有问题的, 根本不会收敛.
    top_k 函数是某种精度函数, 精度不是微分函数. 使用 top_k 无法得到梯度, 权重不能更新.
    https://www.zhihu.com/question/273719492
    https://www.pythonheidong.com/blog/article/301260/

    paper: https://www.aclweb.org/anthology/P14-1062.pdf
    """
    def __init__(self, vocab_size, batch_size, seq_len, embedding_size, n_layers,
                 conv1d_size_list, filters_list, fold_size_list, k_list,
                 k_top, n_classes, name='cnn_kmaxpool_text'):
        self._vocab_size = vocab_size
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._embedding_size = embedding_size
        self._n_layers = n_layers
        self._conv1d_size_list = conv1d_size_list
        self._filters_list = filters_list
        self._fold_size_list = fold_size_list
        self._k_list = k_list
        self._k_top = k_top
        self._n_classes = n_classes
        self._name = name

    def build(self):
        model = Sequential(layers=[
            Embedding(
                input_dim=self._vocab_size,
                output_dim=self._embedding_size
            ),
            DynamicKMaxPooling(
                n_layers=self._n_layers,
                batch_size=self._batch_size,
                seq_len=self._seq_len,
                input_dim=self._embedding_size,
                conv1d_size_list=self._conv1d_size_list,
                filters_list=self._filters_list,
                fold_size_list=self._fold_size_list,
                k_list=self._k_list,
                k_top=self._k_top,
                use_bias=True,
                activation=tf.nn.relu,
                return_sequences=True,
                name='dynamic_fold_k_max_pool'
            ),
            Flatten(),
            Concatenate(axis=-1),
            Dense(
                activation=tf.nn.relu,
                units=self._n_classes * 2,
                use_bias=True,
            ),
            Dense(
                activation=tf.nn.relu,
                units=self._n_classes * 4,
                use_bias=True,
            ),
            Dense(
                activation=tf.nn.relu,
                units=self._n_classes * 4,
                use_bias=True,
            ),
            Dense(
                activation=tf.nn.relu,
                units=self._n_classes,
                use_bias=True,
            ),
            Softmax(),
        ], name=self._name)
        return model




def parser():
    tf.flags.DEFINE_integer("vocab_size", 10, 'vocabulary size. ')

    tf.flags.DEFINE_integer("batch_size", 3, 'batch size. ')
    tf.flags.DEFINE_integer("seq_len", 48, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 128, 'word embedding size, will be use for word embedding, position embedding. ')

    tf.flags.DEFINE_integer("n_classes", 1024, 'layers')

    tf.flags.DEFINE_integer("n_layers", 4, 'layers')
    tf.flags.DEFINE_integer("k_top", 4, 'layers')

    # 48, (1 - 1/4)*48=36, (1 - 2/4)*48=24, (1 - 3/4)*48=12,
    tf.flags.DEFINE_string("conv1d_size_list", '5,3,3,3', 'conv1d kernel size of each layers')
    tf.flags.DEFINE_string("filters_list", '32,32,32,32', 'conv1d kernel filters number of each layers, it should be multiple of corresponding `fold size`. ')
    tf.flags.DEFINE_string("fold_size_list", '2,2,2,2', 'fold size of each layers')
    tf.flags.DEFINE_string("k_list", '36,24,12,6', 'k (k max pool) of each layers. it should less than `seq_len` and decreasing.')

    FLAGS = tf.flags.FLAGS
    FLAGS.conv1d_size_list = list(map(lambda x: int(x), FLAGS.conv1d_size_list.split(',')))
    FLAGS.filters_list = list(map(lambda x: int(x), FLAGS.filters_list.split(',')))
    FLAGS.fold_size_list = list(map(lambda x: int(x), FLAGS.fold_size_list.split(',')))
    FLAGS.k_list = list(map(lambda x: int(x), FLAGS.k_list.split(',')))
    return FLAGS


def demo1():
    FLAGS = parser()

    x = np.random.randint(0, FLAGS.vocab_size, size=(FLAGS.batch_size, FLAGS.seq_len), dtype=np.int)
    inputs = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.seq_len), name='inputs')

    model = Sequential(layers=[
        Embedding(
            input_dim=FLAGS.vocab_size,
            output_dim=FLAGS.embedding_size
        ),
        DynamicKMaxPooling(
            n_layers=FLAGS.n_layers,
            batch_size=FLAGS.batch_size,
            seq_len=FLAGS.seq_len,
            input_dim=FLAGS.embedding_size,
            conv1d_size_list=FLAGS.conv1d_size_list,
            filters_list=FLAGS.filters_list,
            fold_size_list=FLAGS.fold_size_list,
            k_list=FLAGS.k_list,
            k_top=FLAGS.k_top,
            use_bias=True,
            activation=tf.nn.relu,
            return_sequences=True,
            name='dynamic_fold_k_max_pool'
        ),
        Flatten(),
        Concatenate(axis=-1),
        Dense(
            activation=tf.nn.relu,
            units=FLAGS.n_classes * 2,
            use_bias=True,
        ),
        Dense(
            activation=tf.nn.relu,
            units=FLAGS.n_classes * 4,
            use_bias=True,
        ),
        Dense(
            activation=tf.nn.relu,
            units=FLAGS.n_classes * 4,
            use_bias=True,
        ),
        Dense(
            activation=tf.nn.relu,
            units=FLAGS.n_classes,
            use_bias=True,
        ),
        Softmax(),
    ])

    outputs = model.call(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    if isinstance(ret, list):
        for a in ret:
            print(a.shape)
    else:
        print(ret.shape)
    return


if __name__ == '__main__':
    demo1()

