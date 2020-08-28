#!/usr/bin/python3
# -*- coding: utf-8 -*-.
import tensorflow as tf


def train_flags():
    # Data loading parameters
    tf.flags.DEFINE_string("padding_token", "<padding>", 'sentence padding token. ')
    tf.flags.DEFINE_string("out_of_vocab_token", "<oov>", 'sentence out_of_vocab token. ')
    tf.flags.DEFINE_integer("max_len", 200, 'sentence max length. ')

    tf.flags.DEFINE_string("positive_data_file", "./data/ham_100.utf8",
                           "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_file", "./data/spam_100.utf8",
                           "Data source for the negative data.")

    tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")

    # Model hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128,
                            "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,7,13,20,30,64,125", "Comma-spearated filter sizes")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training paramters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

    # Save model.
    tf.flags.DEFINE_string("model_path", "./models", 'The path to save the model (ckpt and other files). ')

    # Parse parameters from commands
    FLAGS = tf.flags.FLAGS
    return FLAGS


def predict_flags():
    # Model path.
    tf.flags.DEFINE_string("model_config_path", "./models/model_config.json", 'The path to save the model (ckpt and other files). ')

    tf.flags.DEFINE_string("positive_data_file", "./data/ham_100.utf8",
                           "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_file", "./data/spam_100.utf8",
                           "Data source for the negative data.")

    # Parse parameters from commands
    FLAGS = tf.flags.FLAGS
    return FLAGS
