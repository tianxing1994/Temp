#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
tensorflow 训练 skip-gram 词向量.

参考链接:
https://blog.csdn.net/just_sort/article/details/87886385

数据下载链接:
http://mattmahoney.net/dc/text8.zip
"""
from collections import Counter, deque
import math
import os
import random
import numpy as np
import tensorflow as tf


class SkipGram(object):
    def __init__(self, fpath, vocabulary_size=5000, window=1, num_skips=2, batch_size=128,
                 embedding_size=128, num_sampled=64, model_path='model/skip_gram.ckpt'):
        self._fpath = fpath
        self._global_index = 0

        assert batch_size % num_skips == 0
        assert num_skips <= 2 * window

        # vocabulary_size: 指定统计词汇表的词汇数量. 超出的不常见词被当作 unk
        self._vocabulary_size = vocabulary_size
        self._window = window
        self._num_skips = num_skips

        # document_idx: list. 长度与 document 相同. 但每个词用其在词汇表中的索引表示.
        self._document_idx = None
        # count: 二元元组的列表. 每个二元元组第一个元素表示词, 对应第二个元素为该词出现的次数.
        self._count = None
        # word_idx: dict. key 为词. value 为分配给该词的索引.
        self._word_idx = None
        # idx_word: dict. key 为词的索引. value 为该索引对应的词.
        self._idx_word = None
        self._build_statistics()

        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._num_sampled = num_sampled
        self._model_path = model_path

        # 最终训练出的 embeding 词向量.
        self._final_embedings = None

    @staticmethod
    def load_document(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            document = f.read().split()
        return document

    def _build_statistics(self):
        """构建文档统计信息."""
        document = self.load_document(self._fpath)
        counter = Counter(document)
        count = [['UNK', -1]]
        count.extend(counter.most_common(self._vocabulary_size - 1))

        word_idx = dict()
        for word, _ in count:
            word_idx[word] = len(word_idx)

        document_idx = list()
        unk_count = 0
        for word in document:
            if word in word_idx:
                idx = word_idx[word]
            else:
                idx = 0
                unk_count += 1
            document_idx.append(idx)

        count[0] = ('unk', unk_count)
        idx_word = dict(zip(word_idx.values(), word_idx.keys()))

        self._document_idx = document_idx
        self._count = count
        self._word_idx = word_idx
        self._idx_word = idx_word
        print(f"DEBUG: statictics init done! ")
        return document_idx, count, word_idx, idx_word

    def generate_batch(self):
        batch = np.ndarray(shape=(self._batch_size,), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        span = 2 * self._window + 1
        buffer = deque(maxlen=span)

        for _ in range(span):
            buffer.append(self._document_idx[self._global_index])
            self._global_index = (self._global_index + 1) % len(self._document_idx)

        for i in range(self._batch_size // self._num_skips):
            target = self._window
            target_to_avoid = [self._window]

            for j in range(self._num_skips):
                while target in target_to_avoid:
                    target = random.randint(0, span - 1)
                target_to_avoid.append(target)
                batch[i * self._num_skips + j] = buffer[self._window]
                labels[i * self._num_skips + j, 0] = buffer[target]
            buffer.append(self._document_idx[self._global_index])
            self._global_index = (self._global_index + 1) % len(self._document_idx)
        return batch, labels

    @staticmethod
    def _net(inputs, vocabulary_size, embedding_size):

        embeddings = tf.Variable(
            tf.random_uniform(shape=(vocabulary_size, embedding_size),
                              minval=-1.0,
                              maxval=1.0)
        )

        embedded = tf.nn.embedding_lookup(embeddings, inputs)
        nce_weights = tf.Variable(tf.truncated_normal(
            (vocabulary_size, embedding_size),
            stddev=1.0/math.sqrt(embedding_size))
        )
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        return embedded, nce_weights, nce_biases, embeddings

    @staticmethod
    def _loss_function(inputs, labels, nce_weights, nce_biases, num_sampled, num_classes):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=labels,
                           inputs=inputs,
                           num_sampled=num_sampled,
                           num_classes=num_classes)
        )
        return loss

    def train(self):
        inputs = tf.placeholder(tf.int32, shape=(self._batch_size,))
        labels = tf.placeholder(tf.int32, shape=(self._batch_size, 1))

        embedded, nce_weights, nce_biases, embeddings = self._net(inputs, self._vocabulary_size, self._embedding_size)
        loss = self._loss_function(embedded, labels, nce_weights, nce_biases, self._num_sampled, self._vocabulary_size)

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        init = tf.global_variables_initializer()

        num_steps = 100001

        saver = tf.train.Saver(max_to_keep=3)

        with tf.Session() as session:
            if self._model_path is not None:
                try:
                    model_dir = os.path.dirname(self._model_path)
                    ckpt = tf.train.latest_checkpoint(model_dir)

                    saver.restore(session, ckpt)
                    print(f"Model restored from file: {ckpt}")
                except ValueError:
                    print(f"Can't not load model. ")

            # init.run()
            session.run(init)

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = self.generate_batch()

                # training
                _, loss_val = session.run(
                    [optimizer, loss],
                    feed_dict={inputs: batch_inputs, labels: batch_labels}
                )
                average_loss += loss_val

                if step % 2000 == 0 and step != 0:
                    average_loss /= 2000
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # validation
                if step % 10000 == 0 and step != 0:
                    valid_size = 16
                    valid_window = self._vocabulary_size
                    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
                    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

                    # embeddings 向量归一化.
                    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
                    normalized_embeddings = embeddings / norm

                    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
                    similarity = tf.matmul(valid_embeddings, normalized_embeddings,
                                           transpose_b=True)
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = self._idx_word[valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                        log_str = "%s nearest to:" % valid_word
                        for k in range(top_k):
                            close_word = self._idx_word[nearest[k]]
                            log_str = "%s %s, " % (log_str, close_word)
                        print(log_str)
                    self._final_embedings = normalized_embeddings.eval()
                    saver.save(session, self._model_path, global_step=step)
        return


def demo1():
    fpath = '../data/word2vec/dc/text8'
    model_path = '../model/word2vec/skipgram/skip_gram.ckpt'
    sg = SkipGram(fpath=fpath, model_path=model_path)
    sg.train()
    print(sg._final_embedings)
    print(sg._final_embedings.shape)
    return


if __name__ == '__main__':
    demo1()
