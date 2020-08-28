#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(pwd)

import json

import tensorflow as tf

from text_cnn import Model, TextCNN
from utils.load_data import LoadData
from utils.parsers import train_flags


def print_flags(flags):
    for k, v in flags.flag_values_dict().items():
        print(f'{k} = {v}')


def save_model(save_dir, model, load_data, model_config):
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
    model_path = os.path.join(ckpt_dir, 'model.ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save(model_path)

    word2id_pkl_path = os.path.join(save_dir, 'word2id.pkl')
    load_data.save_component(word2id_pkl_path)

    model_config.update({
        'model_path': os.path.abspath(model_path),
        'word2id_pkl_path': os.path.abspath(word2id_pkl_path),
        'sentence_max_length': load_data.max_len,
        'padding_token': load_data.padding,
        'out_of_vocab_token': load_data.out_of_vocab,
    })

    model_config_path = os.path.join(save_dir, 'model_config.json')
    with open(model_config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=4)
    return


def train():
    FLAGS = train_flags()
    print_flags(FLAGS)

    load_data = LoadData(
        positive_fpath=FLAGS.positive_data_file,
        negative_fpath=FLAGS.negative_data_file,
        max_len=FLAGS.max_len,
        out_of_vocab=FLAGS.out_of_vocab_token,
        padding=FLAGS.padding_token
    )

    sentences = load_data.padded_id_sentences
    classes = load_data.classes

    model = TextCNN(
        seq_len=FLAGS.max_len,
        embedding_size=FLAGS.embedding_dim,
        n_classes=FLAGS.num_labels,
        vocab_size=load_data.vocab_size,
        filter_sizes=FLAGS.filter_sizes.split(','),
        n_filters=FLAGS.num_filters,
        dropout_keep_prob=FLAGS.dropout_keep_prob,
        l2_reg_lambda=FLAGS.l2_reg_lambda
    ).build()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    loss = Model.loss
    accuracy = Model.accuracy
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    model.fit(x=sentences, y=classes, batch_size=FLAGS.batch_size, epochs=FLAGS.num_epochs)

    model_config = {
        'embedding_dim': FLAGS.embedding_dim,
        'num_labels': FLAGS.num_labels,
        'filter_sizes': FLAGS.filter_sizes,
        'num_filters': FLAGS.num_filters,
        'dropout_keep_prob': FLAGS.dropout_keep_prob,
        'l2_reg_lambda': FLAGS.l2_reg_lambda
    }

    save_model(FLAGS.model_path, model, load_data, model_config)
    return


if __name__ == '__main__':
    train()
