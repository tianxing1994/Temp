#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://nlp.stanford.edu/pubs/glove.pdf
https://github.com/GradySimon/tensorflow-glove
https://github.com/manasRK/glove-gensim
"""
from collections import Counter, defaultdict
from random import shuffle
import tensorflow as tf


class GloVe(object):
    def __init__(self, embedding_size=128, window=3, max_vocab_size=10000, min_occurrences=1,
                 scaling_factor=3/4, x_max=100, batch_size=128, n_epoch=30, learning_rate=0.05):
        self._embedding_size = embedding_size
        self._window = window
        self._max_vocab_size = max_vocab_size
        self._min_occurrences = min_occurrences
        self._scaling_factor = scaling_factor
        self._x_max = x_max
        self._batch_size = batch_size
        self._n_epoch = n_epoch
        self._learning_rate = learning_rate
        self._words = None
        self._vocab_size = None
        self._word2idx = None
        self._cooccurrence_matrix = None
        self._embeddings = None

    def _context_windows(self, region):
        for i, word in enumerate(region):
            start_idx = i - self._window
            end_idx = i + self._window
            left_context = self._context(region, start_idx, i - 1)
            right_context = self._context(region, i + 1, end_idx)
            yield (left_context, word, right_context)

    @staticmethod
    def _context(region, start_idx, end_idx):
        last_idx = len(region)
        selected_tokens = region[max(start_idx, 0): min(end_idx + 1, last_idx)]
        return selected_tokens

    def build_statistics(self, corpus):
        """
        构建语料库的统计信息.
        :param corpus: list(list). 包括列表的列表.
        内层列表代表一句话 sentence, 内层列表中的每一项是一个单词.
        :return:
        """
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for l_context, word, r_context in self._context_windows(region):
                for i, context_word in enumerate(l_context[::-1]):
                    # 使用 1 / (i + 1), 离 word 近的共现词, 权重比较大.
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 1:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generators ?")
        self._words = [word for word, count in word_counts.most_common(self._max_vocab_size)
                       if count >= self._min_occurrences]
        print(f"words init done. words count: {len(self._words)}")
        self._vocab_size = len(self._words)
        self._word2idx = {word: i for i, word in enumerate(self._words)}
        print(f"word to index init done. dict size: {len(self._word2idx)}")
        self._cooccurrence_matrix = {
            (self._word2idx[words[0]], self._word2idx[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self._word2idx and words[1] in self._word2idx
        }
        print(f"cooccurrence matrix init done. matrix size: {len(self._cooccurrence_matrix)}")
        return

    def net(self):
        x_max = tf.constant([self._x_max], dtype=tf.float32, name='x_max')
        scaling_factor = tf.constant([self._scaling_factor], dtype=tf.float32,
                                     name='scaling_factor')
        self._focal_input = tf.placeholder(tf.int32, shape=(None,),
                                           name='focal_words')
        self._context_input = tf.placeholder(tf.int32, shape=(None,),
                                             name='context_words')
        self._cooccurrence_count = tf.placeholder(tf.float32, shape=(None,),
                                                  name='cooccurrence_count')
        embeddings = tf.Variable(
            tf.random_uniform((self._vocab_size, self._embedding_size), 1.0, -1.0),
            name='focal_embeddings'
        )

        focal_embedding = tf.nn.embedding_lookup([embeddings], self._focal_input)
        context_embedding = tf.nn.embedding_lookup([embeddings], self._context_input)

        weighting_factor = tf.minimum(
            1.0,
            tf.pow(tf.div(self._cooccurrence_count, x_max), scaling_factor),
        )

        embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), axis=1)
        log_cooccurrences = tf.log(tf.to_float(self._cooccurrence_count))

        distance_expr = tf.square(tf.add_n([
            embedding_product,
            tf.negative(log_cooccurrences)
        ]))

        single_losses = tf.multiply(weighting_factor, distance_expr)
        total_loss = tf.reduce_mean(single_losses)
        optimizer = tf.train.AdagradOptimizer(self._learning_rate).minimize(total_loss)
        return total_loss, optimizer, embeddings

    def net1(self):
        x_max = tf.constant([self._x_max], dtype=tf.float32, name='x_max')
        scaling_factor = tf.constant([self._scaling_factor], dtype=tf.float32,
                                     name='scaling_factor')
        self._focal_input = tf.placeholder(tf.int32, shape=(None,),
                                           name='focal_words')
        self._context_input = tf.placeholder(tf.int32, shape=(None,),
                                             name='context_words')
        self._cooccurrence_count = tf.placeholder(tf.float32, shape=(None,),
                                                  name='cooccurrence_count')
        focal_embeddings = tf.Variable(
            tf.random_uniform((self._vocab_size, self._embedding_size), 1.0, -1.0),
            name='focal_embeddings'
        )
        context_embeddings = tf.Variable(
            tf.random_uniform((self._vocab_size, self._embedding_size), 1.0, -1.0),
            name='context_embeddings'
        )
        focal_biases = tf.Variable(
            tf.random_uniform((self._vocab_size,), 1.0, -1.0),
            name='focal_biases'
        )
        context_biases = tf.Variable(
            tf.random_uniform((self._vocab_size,), 1.0, -1.0),
            name='context_biases'
        )
        focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self._focal_input)
        context_embedding = tf.nn.embedding_lookup([context_embeddings], self._context_input)
        focal_bias = tf.nn.embedding_lookup([focal_biases], self._focal_input)
        context_bias = tf.nn.embedding_lookup([context_biases], self._context_input)

        weighting_factor = tf.minimum(
            1.0,
            tf.pow(tf.div(self._cooccurrence_count, x_max), scaling_factor),
        )

        embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), axis=1)
        log_cooccurrences = tf.log(tf.to_float(self._cooccurrence_count))

        distance_expr = tf.square(tf.add_n([
            embedding_product,
            focal_bias,
            context_bias,
            tf.negative(log_cooccurrences)
        ]))

        single_losses = tf.multiply(weighting_factor, distance_expr)
        total_loss = tf.reduce_mean(single_losses)
        optimizer = tf.train.AdagradOptimizer(self._learning_rate).minimize(total_loss)
        combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                     name='combined_embeddings')
        return total_loss, optimizer, combined_embeddings

    def batch_generator(self):
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self._cooccurrence_matrix.items()]
        shuffle(cooccurrences)

        for i in range(0, len(cooccurrences), self._batch_size):
            i_indices, j_indices, counts = zip(*cooccurrences[i:i+self._batch_size])
            yield i_indices, j_indices, counts

    def train(self):
        loss, optimizer, embeddings = self.net()

        total_steps = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(f"start to training. ")
            for epoch in range(self._n_epoch):
                for i_indices, j_indices, counts in self.batch_generator():
                    _, curr_loss = sess.run([optimizer, loss], feed_dict={
                        self._focal_input: i_indices,
                        self._context_input: j_indices,
                        self._cooccurrence_count: counts
                    })
                    total_steps += 1
                print(f"current loss: {curr_loss}")
            embeddings = sess.run(embeddings)
        print(embeddings.shape)
        return


def load_document(fpath):
    """
    '../data/word2vec/dc/text8'
    这个文档是所有单词都在一行, 空隔隔开,
    所以此处返回的值是一个列表, 列表中的每一项是一个单词.
    """
    with open(fpath, 'r', encoding='utf-8') as f:
        document = f.read().split()
    return document


def demo1():
    fpath = '../data/word2vec/dc/text8'
    document = load_document(fpath)
    glove = GloVe()
    glove.build_statistics([document])
    glove.train()
    return


if __name__ == '__main__':
    demo1()
