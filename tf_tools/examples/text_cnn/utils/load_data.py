#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import pickle
import re

import numpy as np


class LoadData(object):
    def __init__(self, positive_fpath, negative_fpath, max_len,
                 out_of_vocab='<oov>', padding='<padding>', id2word_pkl_path=None):
        self.positive_fpath = positive_fpath
        self.negative_fpath = negative_fpath
        self.max_len = max_len
        self.out_of_vocab = out_of_vocab
        self.padding = padding
        self.id2word_pkl_path = id2word_pkl_path

        self.sentences = None
        self.classes = None

        self.vocab_size = None
        self.word2id = None
        self.id2word = None

        self.padded_sentences = None
        self.padded_id_sentences = None
        self.init_component()

    def init_component(self):
        sentences, classes = self.load_data()
        self.sentences = sentences
        self.classes = np.array(classes, dtype=np.int32)

        vocab_size, word2id, id2word = self.init_vocabulary()
        self.vocab_size = vocab_size
        self.word2id = word2id
        self.id2word = id2word

        padded_sentences = self.init_padded_sentences(self.sentences)
        self.padded_sentences = padded_sentences

        padded_id_sentences = self.init_id_sentences(self.padded_sentences)
        self.padded_id_sentences = np.array(padded_id_sentences, dtype=np.int32)

    def save_component(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.word2id, f)

    @staticmethod
    def re_han(string):
        pattern = re.compile("[\u4e00-\u9fff]")
        han_list = re.findall(pattern, string)
        return han_list

    def read_zh_file(self, fpath):
        with open(fpath, "r", encoding='utf-8') as f:
            lines = list()
            for line in f:
                line = self.re_han(line)
                lines.append(line)
        return lines

    def init_vocabulary(self):
        if self.id2word_pkl_path is None:
            return self.init_vocabulary_from_sentences(self.sentences)
        else:
            return self.init_vocabulary_from_pkl(self.id2word_pkl_path)

    def init_vocabulary_from_sentences(self, sentences):
        words = list()
        for sentence in sentences:
            words.extend(sentence)
        words = set(words)
        words = list(words)
        vocab_size = len(words)
        ids = np.arange(vocab_size)
        word2id = dict(zip(words, ids))
        id2word = {v: k for k, v in word2id.items()}

        l = len(word2id)
        oov_id = l
        padding_id = l + 1
        word2id[self.out_of_vocab] = oov_id
        word2id[self.padding] = padding_id

        id2word[oov_id] = self.out_of_vocab
        id2word[padding_id] = self.padding
        vocab_size = len(word2id)
        return vocab_size, word2id, id2word

    def init_vocabulary_from_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            word2id = pickle.load(f)
        id2word = {v: k for k, v in word2id.items()}
        vocab_size = len(word2id)
        return vocab_size, word2id, id2word

    def load_data(self):
        pos_sentences = self.read_zh_file(self.positive_fpath)
        neg_sentences = self.read_zh_file(self.negative_fpath)
        sentences = pos_sentences + neg_sentences
        pos_classes = [[1, 0]] * len(pos_sentences)
        neg_classes = [[0, 1]] * len(neg_sentences)
        classes = pos_classes + neg_classes
        return sentences, classes

    def init_padded_sentences(self, sentences):
        padded_sentences = list()
        for sentence in sentences:
            if len(sentence) > self.max_len:
                sentence = sentence[:self.max_len]
                padded_sentences.append(sentence)
            else:
                sentence.extend([self.padding] * (self.max_len - len(sentence)))
                padded_sentences.append(sentence)
        return padded_sentences

    def init_id_sentences(self, sentences):
        s_list = list()
        for sentence in sentences:
            s = list()
            for word in sentence:
                idx = self.word2id.get(word)
                if idx is None:
                    s.append(self.word2id[self.out_of_vocab])
                else:
                    s.append(idx)
            s_list.append(s)
        return s_list


def demo1():
    load_data = LoadData('../data/ham_100.utf8', '../data/spam_100.utf8', max_len=200)
    print(load_data.vocab_size)

    print(load_data.padded_id_sentences)
    print(load_data.classes)
    return


if __name__ == '__main__':
    demo1()
