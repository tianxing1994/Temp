#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://arxiv.org/pdf/1706.03762.pdf
https://github.com/tensorflow/tensor2tensor

https://blog.csdn.net/linchuhai/article/details/90054183
https://segmentfault.com/a/1190000018601461
"""
import numpy as np
import tensorflow as tf


class TransformerConfig(object):
    embedding_size = 512
    num_layers = 6
    keep_prob = 0.9
    learning_rate = 1e-9
    learning_decay_steps = 100
    learning_decay_rate = 0.98
    clip_gradient = 100
    is_embedding_scale = True
    multihead_num = 8
    label_smoothing = 0.1


class Config(object):
    encoder_vocabs = 10000
    decoder_vocabs = 10000
    max_encoder_len = 20
    max_decoder_len = 20
    share_embedding = 10


class Transformer(object):
    def __init__(self,
                 embedding_size=TransformerConfig.embedding_size,
                 num_layers=TransformerConfig.num_layers,
                 keep_prob=TransformerConfig.keep_prob,
                 learning_rate=TransformerConfig.learning_rate,
                 learning_decay_steps=TransformerConfig.learning_decay_steps,
                 learning_decay_rate=TransformerConfig.learning_decay_rate,
                 clip_gradient=TransformerConfig.clip_gradient,
                 is_embedding_scale=TransformerConfig.is_embedding_scale,
                 multihead_num=TransformerConfig.multihead_num,
                 label_smoothing=TransformerConfig.label_smoothing,
                 max_gradient_norm=TransformerConfig.clip_gradient,
                 encoder_vocabs=Config.encoder_vocabs + 2,
                 decoder_vocabs=Config.decoder_vocabs + 2,
                 max_encoder_len=Config.max_encoder_len,
                 max_decoder_len=Config.max_decoder_len,
                 share_embedding=Config.share_embedding,
                 pad_index=None):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.learning_decay_steps = learning_decay_steps
        self.learning_decay_rate = learning_decay_rate
        self.clip_gradient = clip_gradient
        self.is_embedding_scale = is_embedding_scale
        self.multihead_num = multihead_num
        self.label_smoothing = label_smoothing
        self.max_gradient_norm = max_gradient_norm
        self.encoder_vocabs = encoder_vocabs
        self.decoder_vocabs = decoder_vocabs
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.share_embedding = share_embedding
        # [pad] token 所对应的 id.
        self.pad_index = pad_index
        self.build_model()

    def build_model(self):
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, shape=(None,), name='encoder_inputs_length')

        self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='decoder_inputs')
        self.decoder_inputs_length = tf.shape(self.decoder_inputs)[1]

        self.decoder_targets = tf.placeholder(tf.int32, shape=(None, None), name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, shape=(None,), name='decoder_targets_length')

        self.batch_size = tf.placeholder(tf.int32, shape=(None,), name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.targets_mask = tf.sequence_mask(self.decoder_targets_length, self.max_decoder_len, dtype=tf.float32, name='masks')
        self.itf_weight = tf.placeholder(tf.float32, shape=(None, None), name='itf_weight')

        # embedding 层
        with tf.name_scope('embedding'):
            zero = tf.zeros(shape=(1, self.embedding_size), dtype=tf.float32)
            encoder_embedding = tf.get_variable(
                name='embedding_table',
                shape=(self.encoder_vocabs - 1, self.embedding_size),
                initializer=tf.random_normal_initializer(mean=0.0, stddev=self.embedding_size ** -0.5)
            )
            # 将 encoder_embedding 中, [pad] token 所对应的 embedding 的值设置为 0.
            front, end = tf.split(
                value=encoder_embedding,
                num_or_size_splits=(self.pad_index, self.encoder_vocabs - 1 - self.pad_index),
                axis=0
            )
            encoder_embedding = tf.concat(values=(front, zero, end), axis=0)
            # 位置编码
            encoder_position_encoding = self.positional_encoding(self.max_encoder_len)

            inputs_embedding, embedding_mask = self.add_embedding(
                embedding=encoder_embedding,
                inputs_data=self.encoder_inputs,
                position_embedding=encoder_position_encoding
            )

        return

    def positional_encoding(self, sequence_length):
        position_embedding = np.zeros(shape=(sequence_length, self.embedding_size))
        for pos in range(sequence_length):
            for i in range(self.embedding_size // 2):
                position_embedding[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / self.embedding_size))
                position_embedding[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / self.embedding_size))
        position_embedding = tf.convert_to_tensor(position_embedding, dtype=tf.float32)
        return position_embedding

    def add_embedding(self, embedding, inputs_data, position_embedding):
        """将词汇 embedding 与 positional_embedding 相加"""
        inputs_embedding = tf.nn.embedding_lookup(embedding, inputs_data)
        data_length = tf.shape(inputs_embedding)[1]
        inputs_embedding += position_embedding[:data_length, :]
        embedding_mask = tf.expand_dims(
            tf.cast(tf.not_equal(inputs_data, self.pad_index), dtype=tf.float32),
            axis=-1
        )
        inputs_embedding *= embedding_mask
        # inputs_embedding = tf.nn.dropout(inputs_embedding, keep_prob=self.keep_prob)
        return inputs_embedding, embedding_mask

    def multi_head_attention_layer2(self, query, key, score_mask=None, output_mask=None, activation=None, name=None):
        """参考链接中的实现. 感觉不太对. """
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            v = tf.layers.dense(key, units=self.embedding_size, activation=activation, use_bias=False, name='V')
            k = tf.layers.dense(key, units=self.embedding_size, activation=activation, use_bias=False, name='K')
            q = tf.layers.dense(query, units=self.embedding_size, activation=activation, use_bias=False, name='Q')
            v = tf.concat(tf.split(v, self.multihead_num, axis=-1), axis=0)
            k = tf.concat(tf.split(k, self.multihead_num, axis=-1), axis=0)
            q = tf.concat(tf.split(q, self.multihead_num, axis=-1), axis=0)
            # 计算 q, k 的点积, 并进行 scale.
            score = tf.matmul(q, tf.transpose(k, perm=(0, 2, 1)) / tf.sqrt(self.embedding_size / self.multihead_num))
            # mask
            if score_mask is not None:
                score *= score_mask
                score += ((score_mask - 1) * 1e9)
            # softmax
            softmax = tf.nn.softmax(score, dim=2)
            # dropout
            softmax = tf.nn.dropout(softmax, keep_prob=self.keep_prob)
            # attention
            attention = tf.matmul(softmax, v)
            # 将 multi-head 的输出进行拼接.
            concat = tf.concat(tf.split(attention, self.multihead_num, axis=0), axis=-1)
            # linear
            multi_head = tf.layers.dense(concat, units=self.embedding_size,
                                         activation=activation, use_bias=False, name='linear')
            # output mask
            if output_mask is not None:
                multi_head *= output_mask
            # 残差连接前的 dropout
            multi_head = tf.nn.dropout(multi_head, keep_prob=self.keep_prob)
            # 残差链接
            multi_head += query
            # layer norm
            multi_head = tf.contrib.layers.layer_norm(multi_head, begin_norm_axis=2)
        return multi_head

    def scaled_dot_product_attention(self, query, key, value, query_real_len, key_value_real_len, attention_mask):
        """
        Scaled Dot-Product Attention
        :param query: tensor, shape=(batch, query_len, dk)
        :param key: tensor, shape=(batch, key_value_len, dk)
        :param value: tensor, shape=(batch, key_value_len, dv)
        :param query_real_len:
        :param key_value_real_len:
        :param attention_mask:
        :return:
        """
        _, query_len, _ = tf.shape(query)
        _, key_value_len, _ = tf.shape(key)

        ones = tf.ones(shape=(query_real_len, key_value_real_len), dtype=tf.int32)
        paddings = tf.constant(
            value=[[0, query_len - query_real_len],
                   [0, key_value_len - key_value_real_len]],
            dtype=tf.int32)
        score_mask = tf.pad(ones, paddings=paddings, mode="CONSTANT", constant_values=0)

        # shape=(batch, query_len, key_value_len)
        score = tf.matmul(query, tf.transpose(key, perm=(0, 2, 1))) / tf.sqrt(self.embedding_size)
        if score_mask is not None:
            score *= score_mask
            score += ((score_mask - 1) * 1e9)
        score_softmax = tf.nn.softmax(score, axis=-1)
        score_softmax *= attention_mask

        # shape=(batch, query_len, dv)
        attention = tf.matmul(score_softmax, value)
        return attention

    def multi_head_attention(self, query, key, value, dk, dv, real_len, attention_mask,
                                   activation=None, name=None):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            heads = list()
            for i in range(self.multihead_num):
                q = tf.layers.dense(query, units=dk, activation=None, use_bias=False, name=f'q_{i}')
                k = tf.layers.dense(key, units=dk, activation=None, use_bias=False, name=f'k_{i}')
                v = tf.layers.dense(value, units=dv, activation=None, use_bias=False, name=f'v_{i}')

                head = self.scaled_dot_product_attention(
                    query=q, key=k, value=v, query_real_len=real_len,
                    key_value_real_len=real_len, attention_mask=attention_mask
                )
                heads.append(head)
            heads_concat = tf.concat(values=heads, axis=-1)
            result = tf.layers.dense(heads_concat, units=self.embedding_size, activation=None, use_bias=False)
        return result

    def feed_forward_layer(self, inputs, output_mask=None, activation=None, name=None):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            inner_layer = tf.layers.dense(inputs=inputs,
                                          units=4 * self.embedding_size,
                                          activation=activation)
            dense = tf.layers.dense(inputs=inner_layer, units=self.embedding_size, activation=None)
            if output_mask is not None:
                dense *= output_mask
            # dropout
            dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)

            # 残差连接
            dense += inputs

            # Layer Norm
            dense = tf.contrib.layers.layer_norm(dense, begin_norm_axis=2)
        return dense


if __name__ == '__main__':
    pass




































