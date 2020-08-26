#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.keras.layers.embedding import Embedding
from tf_tools.keras.layers.conv1d import Conv1D
from tf_tools.debug_tools.common import session_run


class FoldKMaxPooling(object):
    """
    这个实现是有问题的, 根本不会收敛.
    top_k 函数是某种精度函数, 精度不是微分函数. 使用 top_k 无法得到梯度, 权重不能更新.
    https://www.zhihu.com/question/273719492
    https://www.pythonheidong.com/blog/article/301260/

    paper: https://www.aclweb.org/anthology/P14-1062.pdf
    """
    def __init__(self, batch_size, k=1, fold_size=2, use_bias=True,
                 activation=None, name='fold_k_max_pool'):
        self._batch_size = batch_size
        self._k = k
        self._fold_size = fold_size
        self._use_bias = use_bias
        self._activation = activation
        self._name_scope = tf.name_scope(name)

        self._bias = None

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs

            n = x.shape[-1]._value
            x_split = tf.split(x, num_or_size_splits=n, axis=-1)
            l = len(x_split)

            assert l % self._fold_size == 0

            k_max_list = list()
            for i in range(0, l, self._fold_size):
                with tf.name_scope('fold'):
                    fold = tf.concat(x_split[i: i+self._fold_size], axis=-1)
                    fold = tf.reduce_sum(fold, axis=-1)

                with tf.name_scope('k_max_pool'):
                    x = tf.nn.top_k(fold, self._k, sorted=False).values
                    x = tf.expand_dims(x, axis=-1)
                    k_max_list.append(x)
            x = tf.concat(k_max_list, axis=-1)
            if self._activation is not None:
                x = self._activation(x)
        return x

    def gather(self, params, indices, axis):
        with tf.name_scope('gather'):
            x_split = tf.unstack(params, num=self._batch_size, axis=0)
            i_split = tf.unstack(indices, num=self._batch_size, axis=0)
            batch_x = list()
            for x, i in zip(x_split, i_split):
                i = tf.cast(i, dtype=tf.int32)
                x = tf.gather(params=x, indices=i, axis=axis)
                batch_x.append(x)
            x = tf.stack(batch_x, axis=0)
        return x


class FoldKMaxPoolingV2(object):
    """
    这个实现是有问题的, 根本不会收敛.
    top_k 函数是某种精度函数, 精度不是微分函数. 使用 top_k 无法得到梯度, 权重不能更新.
    https://www.zhihu.com/question/273719492
    https://www.pythonheidong.com/blog/article/301260/

    paper: https://www.aclweb.org/anthology/P14-1062.pdf
    """
    def __init__(self, batch_size, k=1, fold_size=2, use_bias=True,
                 activation=None, name='fold_k_max_pool'):
        self._batch_size = batch_size
        self._k = k
        self._fold_size = fold_size
        self._use_bias = use_bias
        self._activation = activation
        self._name_scope = tf.name_scope(name)

        self._bias = None

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs

            n = x.shape[-1]._value
            x_split = tf.split(x, num_or_size_splits=n, axis=-1)
            l = len(x_split)

            assert l % self._fold_size == 0

            k_max_list = list()
            for i in range(0, l, self._fold_size):
                with tf.name_scope('fold'):
                    fold = tf.concat(x_split[i: i+self._fold_size], axis=-1)
                    fold = tf.reduce_sum(fold, axis=-1)

                with tf.name_scope('k_max_pool'):
                    idx = tf.nn.top_k(fold, self._k, sorted=False).indices
                    s_idx = tf.sort(idx, axis=-1)

                    # 执行 tf.sort 之后, 把形状信息弄丢了, 这里重新给它赋值回去.
                    for i in range(1, len(s_idx.shape)):
                        s_idx.shape[i]._value = idx.shape[i]._value

                    x = self.gather(params=fold, indices=idx, axis=-1)
                    x = tf.expand_dims(x, axis=-1)
                    k_max_list.append(x)
            x = tf.concat(k_max_list, axis=-1)
            if self._activation is not None:
                x = self._activation(x)
        return x

    def gather(self, params, indices, axis):
        with tf.name_scope('gather'):
            x_split = tf.unstack(params, num=self._batch_size, axis=0)
            i_split = tf.unstack(indices, num=self._batch_size, axis=0)
            batch_x = list()
            for x, i in zip(x_split, i_split):
                i = tf.cast(i, dtype=tf.int32)
                x = tf.gather(params=x, indices=i, axis=axis)
                batch_x.append(x)
            x = tf.stack(batch_x, axis=0)
        return x


class DynamicKMaxPooling(object):
    """
    这个实现是有问题的, 根本不会收敛.
    top_k 函数是某种精度函数, 精度不是微分函数. 使用 top_k 无法得到梯度, 权重不能更新.
    https://www.zhihu.com/question/273719492
    https://www.pythonheidong.com/blog/article/301260/

    paper: https://www.aclweb.org/anthology/P14-1062.pdf
    """
    def __init__(self, n_layers, batch_size, seq_len, input_dim, conv1d_size_list, filters_list,
                 fold_size_list, k_list=None, k_top=1, use_bias=True, activation=None,
                 return_sequences=False, name='dynamic_fold_k_max_pool'):
        """
        :param n_layers:
        :param batch_size:
        :param seq_len:
        :param input_dim:
        :param conv1d_size_list: 请确保 conv1d_size_list 中的值与模型运算相匹配.
        :param filters_list: 请确保 filters_list 中的值与模型运算相匹配.
        :param fold_size_list: 请确保 fold_size_list 中的值与模型运算相匹配.
        :param k_list: 论文中的动态 kmaxpooling 是认为句子的长度不同,
        模型根据句子的长度来决定 k 值, 但有的时候句子长度是固定的, 且动态的 k 值不够直观.
        请确保 fold_size_list 中的值与模型运算相匹配.
        :param k_top:
        :param use_bias:
        :param activation:
        :param return_sequences: 如果需要返回每一层的结果, 将其设置为 True,
        返回一个包含每一层的结果的列表. 默认为 False, 只返回最后一层的结果.
        :param name:
        """
        self._n_layers = n_layers
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._input_dim = input_dim

        self._conv1d_size_list = conv1d_size_list
        self._filters_list = filters_list
        self._fold_size_list = fold_size_list
        self._k_list = k_list
        self._k_top = k_top
        self._use_bias = use_bias
        self._activation = activation
        self._return_sequences = return_sequences
        self._name_scope = tf.name_scope(name)

        self._k_max_pool_layers = list()
        self._conv1d_layers = list()

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs
            x_list = list()
            for i in range(self._n_layers):
                if len(self._conv1d_layers) < i+1:
                    filters = self._filters_list[i]
                    conv1d = Conv1D(
                        filters=filters,
                        kernel_size=3, strides=1,
                        padding='full',
                        activation=self._activation,
                        name='conv1d_{}'.format(i+1)
                    )
                    self._conv1d_layers.append(conv1d)

                if len(self._k_max_pool_layers) < i+1:
                    if self._k_list is not None:
                        k = self._k_list[i]
                    else:
                        k = self._dynamic_k_v2(i+1)
                    fold_size = self._fold_size_list[i]
                    fold_k_max_pool = FoldKMaxPooling(
                        batch_size=self._batch_size,
                        k=k,
                        fold_size=fold_size,
                        activation=self._activation,
                        name='fold_k_max_pool_{}'.format(i + 1)
                    )
                    self._k_max_pool_layers.append(fold_k_max_pool)

                x = self._conv1d_layers[i](x)
                x = self._k_max_pool_layers[i](x)
                x_list.append(x)
        if self._return_sequences:
            return x_list
        else:
            return x

    def _dynamic_k(self, i_layer):
        k_calc = int(round((1 - i_layer / self._n_layers) * self._seq_len))
        k = int(max(self._k_top, k_calc))
        return k

    def _dynamic_k_v2(self, i_layer):
        if i_layer == self._n_layers:
            return self._k_top
        else:
            k = int(round((self._seq_len - self._k_top) / self._n_layers * (self._n_layers - i_layer)))
            return k


def parser():
    tf.flags.DEFINE_integer("vocab_size", 10, 'vocabulary size. ')

    tf.flags.DEFINE_integer("batch_size", 3, 'batch size. ')
    tf.flags.DEFINE_integer("seq_len", 48, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 128, 'word embedding size, will be use for word embedding, position embedding. ')

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
    x = np.array(
        [[[1, 1, 1, 1],
          [2, 1, 3, 1],
          [1, 4, 1, 1],
          [1, 1, 1, 2],
          [4, 3, 1, 1]],

         [[2, 2, 2, 2],
          [4, 2, 1, 4],
          [2, 4, 2, 2],
          [3, 2, 2, 2],
          [2, 2, 5, 2]]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')
    fold_k_max_pooling = FoldKMaxPooling(batch_size=2, k=3, fold_size=1)
    # fold_k_max_pooling = FoldKMaxPoolingV2(batch_size=2, k=3, fold_size=1)

    outputs = fold_k_max_pooling(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


def demo2():
    FLAGS = parser()

    x = np.random.randint(0, FLAGS.vocab_size, size=(FLAGS.batch_size, FLAGS.seq_len), dtype=np.int)

    inputs = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.seq_len), name='inputs')

    outputs = Embedding(
        input_dim=10,
        output_dim=FLAGS.embedding_size
    )(inputs)

    outputs = DynamicKMaxPooling(
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
    )(outputs)

    ret = session_run(outputs, feed_dict={inputs: x})

    if isinstance(ret, list):
        for a in ret:
            print(a.shape)
    else:
        print(ret.shape)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
