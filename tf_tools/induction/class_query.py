#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class ClassQueryBlock(object):
    def __init__(self, encode_size, name='class_query_block', reuse=tf.AUTO_REUSE):
        self._encode_size = encode_size
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, class_vector, query_encoder):
        """
        :param class_vector: tensor, shape=(num_classes, encode_size)
        :param query_encoder: tensor, shape=(None, encode_size)
        :return:
        """
        with self.variable_scope:
            with self.name_scope:
                m = tf.get_variable(
                    name='m',
                    shape=(self._encode_size, self._encode_size),
                    dtype=tf.float32
                )

                cv_trans = tf.matmul(class_vector, m)
                score = tf.matmul(query_encoder, cv_trans, transpose_b=True)
                score = tf.nn.relu(score)
        return score


class ClassQuery(object):
    def __init__(self, encode_size, hidden_layers=1, name='class_query', reuse=tf.AUTO_REUSE):
        self._encode_size = encode_size
        self._hidden_layers = hidden_layers
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, class_vector, query_encoder):
        """
        计算查询向量与类向量之产蝗内积 |a||b|cos.
        因为: 类向量已被缩放至模长为 1 (|b|=1); 所有查询结果的查询向量大小不变 (|a| 不变). 所以最终的结果是比较 cos 余弦相似度.
        :param class_vector: tensor, shape=(num_classes, encode_size). 类向量, 输入时, 应确保每个类向量已被缩放至模长为 1.
        :param query_encoder: tensor, shape=(None, encode_size). 查询向量.
        :return:
        """
        with self.variable_scope:
            with self.name_scope:

                num_classes = tf.shape(class_vector)[0]
                score_list = list()
                for i in range(self._hidden_layers):
                    name = f'class_query_block_{i+1}'
                    class_query_block = ClassQueryBlock(
                        encode_size=self._encode_size,
                        name=name
                    )
                    score = class_query_block(class_vector, query_encoder)
                    score_list.append(score)
                score_list = tf.concat(score_list, axis=1)
                score_list = tf.reshape(score_list, shape=(-1, self._hidden_layers, num_classes))
                score_list = tf.transpose(score_list, perm=(0, 2, 1))
                relation_w = tf.get_variable(
                    name='relation_w',
                    shape=(self._hidden_layers, 1),
                    dtype=tf.float32
                )

                relation_b = tf.get_variable(
                    name='relation_b',
                    shape=(1,),
                    dtype=tf.float32
                )

                score_final = tf.matmul(score_list, relation_w) + relation_b
                score_final = tf.nn.softmax(tf.squeeze(score_final, axis=-1), axis=-1)
                # score_final = tf.nn.sigmoid(tf.squeeze(score_final, axis=-1))
        return score_final


def demo1():
    x = np.array(
        [[1, 2, 3, 4], [4, 3, 2, 1], [3, 2, 2, 3], [3, 4, 4, 3], [1, 3, 2, 1]],
        dtype=np.float
    )

    y = np.array(
        [[6, 2, 5, 4], [4, 3, 2, 1], [3, 4, 4, 3], [3, 2, 2, 3], [1, 3, 2, 1]],
        dtype=np.float
    )

    class_vector = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='class_vector')
    query_encoder = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='query_encoder')
    class_query = ClassQuery(
        encode_size=4,
        hidden_layers=2
    )
    outputs = class_query(
        class_vector=class_vector,
        query_encoder=query_encoder
    )

    ret = session_run(outputs, feed_dict={class_vector: x, query_encoder: y})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
