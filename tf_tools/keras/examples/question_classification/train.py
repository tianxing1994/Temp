#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os

import tensorflow as tf

from tf_tools.keras.losses.categorical_crossentropy import CategoricalCrossentropy
from tf_tools.keras.metrics.categorical_accuracy import CategoricalAccuracy
from tf_tools.keras.examples.question_classification.load_data import LoadData
from tf_tools.keras.modeling.cnn_k_max_pool_text import CNNKMaxPoolTextModel
from tf_tools.keras.modeling.cnn_text import CNNTextModel


def train_flags_cnn_k_max_pool_text_model():
    tf.flags.DEFINE_integer("epochs", 200, 'epochs. ')
    tf.flags.DEFINE_float("learning_rate", 1e-4, 'learning rate. ')

    tf.flags.DEFINE_integer("batch_size", 64, 'batch size. ')
    tf.flags.DEFINE_integer("seq_len", 64, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 128, 'word embedding size, will be use for word embedding, position embedding. ')

    tf.flags.DEFINE_integer("n_layers", 4, 'layers')
    tf.flags.DEFINE_integer("k_top", 4, 'layers')

    # 48, (1 - 1/4)*48=36, (1 - 2/4)*48=24, (1 - 3/4)*48=12,
    tf.flags.DEFINE_string("conv1d_size_list", '5,3,3,3', 'conv1d kernel size of each layers')
    tf.flags.DEFINE_string("filters_list", '32,32,32,32', 'conv1d kernel filters number of each layers, it should be multiple of corresponding `fold size`. ')
    tf.flags.DEFINE_string("fold_size_list", '2,2,2,2', 'fold size of each layers')
    tf.flags.DEFINE_string("k_list", '36,24,12,6', 'k (k max pool) of each layers. it should less than `seq_len` and decreasing.')

    # Save model.
    tf.flags.DEFINE_string("model_path", "./models", 'The path to save the model (ckpt and other files). ')

    FLAGS = tf.flags.FLAGS
    FLAGS.conv1d_size_list = list(map(lambda x: int(x), FLAGS.conv1d_size_list.split(',')))
    FLAGS.filters_list = list(map(lambda x: int(x), FLAGS.filters_list.split(',')))
    FLAGS.fold_size_list = list(map(lambda x: int(x), FLAGS.fold_size_list.split(',')))
    FLAGS.k_list = list(map(lambda x: int(x), FLAGS.k_list.split(',')))
    return FLAGS


def train_flags_cnn_text_model():
    tf.flags.DEFINE_integer("epochs", 200, 'epochs. ')
    tf.flags.DEFINE_float("learning_rate", 1e-3, 'learning rate. ')

    tf.flags.DEFINE_integer("batch_size", 64, 'batch size. ')
    tf.flags.DEFINE_integer("seq_len", 12, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 64, 'word embedding size, will be use for word embedding, position embedding. ')

    # 48;
    # (None, 50, 32), (None, 52, 32), (None, 26, 32);
    # (None, 28, 64), (None, 30, 64), (None, 15, 64);
    # (None, 17, 128), (None, 19, 128), (None, 9, 128);
    # (None, 11, 256), (None, 13, 256), (None, 6, 256);
    # tf.flags.DEFINE_string("conv1d_size_list", '3,3,3,3', 'conv1d kernel size of each layers')
    # tf.flags.DEFINE_string("filters_list", '32,64,128,256', 'conv1d kernel filters number of each layers, it should be multiple of corresponding `fold size`. ')
    # tf.flags.DEFINE_string("pool_size_list", '2,2,2,2', 'fold size of each layers')
    # tf.flags.DEFINE_string("feed_forward_units_list", '512,1024,512,256', 'k (k max pool) of each layers. it should less than `seq_len` and decreasing.')

    tf.flags.DEFINE_string("conv1d_size_list", '5,5,5', 'conv1d kernel size of each layers')
    tf.flags.DEFINE_string("filters_list", '32,32,32', 'conv1d kernel filters number of each layers, it should be multiple of corresponding `fold size`. ')
    tf.flags.DEFINE_string("pool_size_list", '2,2,2', 'fold size of each layers')
    tf.flags.DEFINE_string("feed_forward_units_list", '512,1024,512', 'k (k max pool) of each layers. it should less than `seq_len` and decreasing.')

    FLAGS = tf.flags.FLAGS
    FLAGS.conv1d_size_list = list(map(lambda x: int(x), FLAGS.conv1d_size_list.split(',')))
    FLAGS.filters_list = list(map(lambda x: int(x), FLAGS.filters_list.split(',')))
    FLAGS.pool_size_list = list(map(lambda x: int(x), FLAGS.pool_size_list.split(',')))
    FLAGS.feed_forward_units_list = list(map(lambda x: int(x), FLAGS.feed_forward_units_list.split(',')))
    return FLAGS


def print_flags(flags):
    for k, v in flags.flag_values_dict().items():
        print(f'{k} = {v}')


def save_model(FLAGS, model, load_data):
    model_path = os.path.abspath(FLAGS.model_path)
    model_config = {
        'epochs': FLAGS.epochs,
        'learning_rate': FLAGS.learning_rate,
        'batch_size': FLAGS.batch_size,
        'seq_len': FLAGS.seq_len,
        'embedding_size': FLAGS.embedding_size,
        'n_layers': FLAGS.n_layers,
        'k_top': FLAGS.k_top,
        'conv1d_size_list': FLAGS.conv1d_size_list,
        'filters_list': FLAGS.filters_list,
        'fold_size_list': FLAGS.fold_size_list,
        'k_list': FLAGS.k_list,
        'model_path': model_path,
    }

    ckpt_dir = os.path.join(model_path, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
    model.save(ckpt_path)

    load_data_config = load_data.save_component(model_path)

    model_config.update(load_data_config)
    model_config_path = os.path.join(model_path, 'model_config.json')
    with open(model_config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=4)
    return


def train():
    FLAGS = train_flags_cnn_text_model()
    print_flags(FLAGS)

    load_data = LoadData(max_len=FLAGS.seq_len)
    sentences = load_data.sentences
    classes = load_data.classes
    vocab_size = load_data.vocabulary_obj.vocab_size
    n_classes = load_data.classes_obj.n_classes

    # paper: https://www.aclweb.org/anthology/P14-1062.pdf
    # model = CNNKMaxPoolTextModel(
    #     vocab_size=vocab_size,
    #     batch_size=FLAGS.batch_size,
    #     seq_len=FLAGS.seq_len,
    #     embedding_size=FLAGS.embedding_size,
    #     n_layers=FLAGS.n_layers,
    #     conv1d_size_list=FLAGS.conv1d_size_list,
    #     filters_list=FLAGS.filters_list,
    #     fold_size_list=FLAGS.fold_size_list,
    #     k_list=FLAGS.k_list,
    #     k_top=FLAGS.k_top,
    #     n_classes=n_classes
    # ).build()

    model = CNNTextModel(
        vocab_size=vocab_size,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
        embedding_size=FLAGS.embedding_size,
        conv1d_size_list=FLAGS.conv1d_size_list,
        filters_list=FLAGS.filters_list,
        pool_size_list=FLAGS.pool_size_list,
        feed_forward_units_list=FLAGS.feed_forward_units_list,
        n_classes=n_classes,
        name='cnn_text'
    ).build()

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy(), ]
    )

    model.fit(
        x=sentences,
        y=classes,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs
    )

    save_model(FLAGS, model, load_data)
    return


if __name__ == '__main__':
    train()
