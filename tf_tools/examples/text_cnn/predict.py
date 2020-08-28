#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(pwd)

import json

import numpy as np

from text_cnn import TextCNN
from utils.load_data import LoadData
from utils.parsers import predict_flags


def print_flags(flags):
    for k, v in flags.flag_values_dict().items():
        print(f'{k} = {v}')


def save_model(save_dir, model, load_data):
    """
    需要存储:
    1. .ckpt 文件.
    2. word2id.pkl
    3. model_config.json: {
        'model_path': 'ckpt/',
        'sentence_max_length': 200,
        'padding_token': '<padding>',
        'out_of_vocab_token': '<oov>',
    }
    :return:
    """
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    model_path = os.path.join(ckpt_dir, 'model')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save(model_path)

    word2id_pkl = os.path.join(save_dir, 'word2id.pkl')
    load_data.save_component(word2id_pkl)

    model_config = {
        'model_path': ckpt_dir,
        'sentence_max_length': load_data.max_len,
        'padding_token': load_data.padding,
        'out_of_vocab_token': load_data.out_of_vocab,
    }

    model_config_path = os.path.join(save_dir, 'model_config.json')
    with open(model_config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=4)
    return


def predict():
    FLAGS = predict_flags()
    print_flags(FLAGS)

    model_config_path = FLAGS.model_config_path
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    load_data = LoadData(
        positive_fpath=FLAGS.positive_data_file,
        negative_fpath=FLAGS.negative_data_file,
        max_len=model_config['sentence_max_length'],
        out_of_vocab=model_config['out_of_vocab_token'],
        padding=model_config['padding_token'],
        id2word_pkl_path=model_config['word2id_pkl_path']
    )

    sentences = load_data.padded_id_sentences
    classes = load_data.classes

    model = TextCNN(
        seq_len=model_config['sentence_max_length'],
        embedding_size=model_config['embedding_dim'],
        n_classes=model_config['num_labels'],
        vocab_size=load_data.vocab_size,
        filter_sizes=model_config['filter_sizes'].split(','),
        n_filters=model_config['num_filters'],
        dropout_keep_prob=model_config['dropout_keep_prob'],
        l2_reg_lambda=model_config['l2_reg_lambda']
    ).build()

    model.restore(model_config['model_path'])

    y_pred = model.predict(sentences)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(classes, axis=-1)

    accuracy = np.mean(np.array(np.equal(y_pred, y_true), dtype=np.float32))
    print(f'accuracy: {accuracy}')
    return


if __name__ == '__main__':
    predict()
