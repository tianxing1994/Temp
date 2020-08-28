#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import Counter
import os
import pickle

import numpy as np

from tf_tools.tokenizer.delimiter_tokenizer import DelimiterTokenizer


class Vocabulary(object):
    def __init__(self, max_len, min_word_freq=1, max_vocab=None,
                 tokenizer=None, padding='<pad>', out_of_vocab='<oov>'):
        """
        :param min_word_freq:
        :param max_vocab:
        key is the name of word, which will be set as attr name to self. value is the word.
        :param tokenizer:
        """
        self._max_len = max_len
        self._min_word_freq = min_word_freq
        self._max_vocab = max_vocab
        self.tokenizer = tokenizer

        self.padding = padding
        self.out_of_vocab = out_of_vocab

        self.token_sentences = None

        self.words2ids = None
        self.ids2words = None
        self.vocab_size = None

    def init_vocab_from_sentences(self, sentences):
        """
        初始化词汇表.
        因为初始化词汇表, 和获得 id 化的句子, 可能不是在同一个数据集中, 所以将这两部分分开.
        """
        token_sentences = self._tokenize(sentences)
        self.token_sentences = token_sentences
        self._build_vocab(token_sentences)

    def sentences_to_ids(self, sentences: list):
        """将句子转化为 id. 使用此方法之前, 请确定已初始化词汇表. """
        token_sentences = self._tokenize(sentences)
        sentences_pad_id = list()
        for sentence in token_sentences:
            sentence_pad = self._pad_or_truncate_sentence(sentence)
            sentence_pad_id = self._sentence_to_id(sentence_pad)
            sentences_pad_id.append(sentence_pad_id)
        result = np.array(sentences_pad_id)
        return result

    def _tokenize(self, sentences):
        """
        :param sentences: list of str.
        :return:
        """
        result = list()
        for sentence in sentences:
            w_list = self.tokenizer.tokenize(sentence)
            result.append(w_list)
        return result

    def _build_vocab(self, token_sentences):
        counter = Counter()
        for token_sentence in token_sentences:
            counter.update(token_sentence)

        counter = Counter(dict(filter(lambda x: x[1] >= self._min_word_freq, counter.items())))
        if self._max_vocab is not None:
            counter = Counter(dict(counter.most_common(n=self._max_vocab)))
        words = list(counter.keys())
        words.append(self.padding)
        words.append(self.out_of_vocab)

        ids = list(np.arange(len(words)))
        words2ids = dict(zip(words, ids))
        ids2words = dict(zip(ids, words))

        self.words2ids = words2ids
        self.ids2words = ids2words
        self.vocab_size = len(self.words2ids)

    def _pad_or_truncate_sentence(self, sentence: list):
        l = len(sentence)
        if l > self._max_len:
            sentence = sentence[:self._max_len]
        else:
            sentence = sentence + [self.padding] * (self._max_len - l)
        return sentence

    def _sentence_to_id(self, sentence: list):
        ids = list()
        oov_id = self.words2ids[self.padding]
        for token in sentence:
            ids.append(self.words2ids.get(token, oov_id))
        return ids

    def get_config(self):
        ret = {
            'max_len': self._max_len,
            'min_word_freq': self._min_word_freq,
            'max_vocab': self._max_vocab,
            'padding': self.padding,
            'out_of_vocab': self.out_of_vocab,
            'vocab_size': self.vocab_size
        }

        ret.update(self.tokenizer.get_config())
        return ret

    def save_component(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        fpath = os.path.join(path, 'words2ids.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.words2ids, f)
        return fpath


class Classes(object):
    def __init__(self, out_of_classes='<oov>'):
        self._out_of_classes = out_of_classes
        self.n_classes = None
        self.classes2ids = None
        self.ids2classes = None

    def init_from_classes(self, classes):
        unique_classes = list(set(classes))
        unique_classes.append(self._out_of_classes)
        self.n_classes = len(unique_classes)

        ids = list(np.arange(self.n_classes))

        self.classes2ids = dict(zip(unique_classes, ids))
        self.ids2classes = dict(zip(ids, unique_classes))

    def classes_to_one_hot(self, classes):
        ids = self.classes_to_ids(classes)
        result = self.one_hot(ids)
        return result

    def classes_to_ids(self, classes):
        oov_id = self.classes2ids[self._out_of_classes]
        result = list()
        for cls in classes:
            idx = self.classes2ids.get(cls, oov_id)
            result.append(idx)
        return result

    def one_hot(self, idx):
        if isinstance(idx, list):
            result = list()
            for i in idx:
                one_hot = [0] * self.n_classes
                one_hot[i] = 1
                result.append(one_hot)
        else:
            result = [0] * self.n_classes
            result[idx] = 1
        result = np.array(result)
        return result

    def get_config(self):
        ret = {
            'out_of_classes': self._out_of_classes,
            'n_classes': self.n_classes
        }
        return ret

    def save_component(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        fpath = os.path.join(path, 'classes2ids.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.classes2ids, f)
        return fpath


def demo1():
    sentences = [
        'What films featured the character Popeye Doyle ?',
        'What is the temperature for cooking ?'
    ]
    vocabulary = Vocabulary(
        max_len=10,
        tokenizer=DelimiterTokenizer(sep=' ')
    )
    vocabulary.init_vocab_from_sentences(sentences)
    ret = vocabulary.sentences_to_ids(sentences)
    print(ret)
    return


if __name__ == '__main__':
    demo1()
