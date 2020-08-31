#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf

from config import project_path
from tf_tools.keras.losses import SparseCategoricalCrossentropy
from tf_tools.keras.metrics import SparseCategoricalAccuracy
from tf_tools.keras.modeling.bert_vector_classify import BertVectorClassify
from tf_tools.data.text_classification.base_dataset import TwoColumnsCSVDataSet
from tf_tools.data.nlp_encoder import LabelsEncoder
from tf_tools.bert.pre_trained.bert_vector import BertVector


def get_flags():
    vocab_file = os.path.join(project_path, 'tf_tools/bert/pre_trained/chinese_L-12_H-768_A-12/vocab.txt')
    bert_config_file = os.path.join(project_path, 'tf_tools/bert/pre_trained/chinese_L-12_H-768_A-12/bert_config.json')
    init_checkpoint = os.path.join(project_path, 'tf_tools/bert/pre_trained/chinese_L-12_H-768_A-12/bert_model.ckpt')

    tf.flags.DEFINE_string("layer_indexes", "-1,-2,-3,-4", "")

    tf.flags.DEFINE_string(
        "bert_config", bert_config_file,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    tf.flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

    tf.flags.DEFINE_string(
        "init_checkpoint", init_checkpoint,
        "Initial checkpoint (usually from a pre-trained BERT model).")

    tf.flags.DEFINE_string("vocab_file", vocab_file,
                        "The vocabulary file that the BERT model was trained on.")

    tf.flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

    tf.flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

    tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

    tf.flags.DEFINE_string("master", None,
                        "If using a TPU, the address of the master.")

    tf.flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")

    tf.flags.DEFINE_bool(
        "use_one_hot_embeddings", False,
        "If True, tf.one_hot will be used for embedding lookups, otherwise "
        "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
        "since it is much faster.")

    FLAGS = tf.flags.FLAGS

    from tf_tools.bert.pre_trained import modeling
    FLAGS.layer_indexes = [int(x) for x in FLAGS.layer_indexes.split(",")]
    FLAGS.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config)
    return FLAGS


def demo1():
    inputs = np.random.random(size=(1000, 768))
    clusters_init_value = np.random.random(size=(10, 768))
    labels = np.random.randint(low=0, high=10, size=(1000,))

    model = BertVectorClassify(
        clusters_init_value=clusters_init_value,
        units_list=[1024, 2048, 2048, 1024, 768]
    ).build()

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy(), ]
    )

    model.fit(x=inputs, y=labels)
    return


def demo2():
    fpath = os.path.join(project_path, 'tf_tools/data/dataset/others/question_classification_dataset.csv')
    dataset = TwoColumnsCSVDataSet(
        csv_path=fpath,
        sentence_col_name='Questions',
        labels_col_name='Category0'
    )
    labels_encoder = LabelsEncoder(label_data_or_pkl=dataset.labels)
    labels_id = labels_encoder.labels_to_ids(dataset.labels)

    ids2labels = labels_encoder.ids2labels.items()
    ids2labels = sorted(ids2labels, key=lambda x: x[0])

    labels_sentences = list(map(lambda x: x[1], ids2labels))

    FLAGS = get_flags()

    bert_vector = BertVector(
        batch_size=FLAGS.batch_size,
        max_seq_length=FLAGS.max_seq_length,
        vocab_file=FLAGS.vocab_file,
        bert_config=FLAGS.bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=FLAGS.layer_indexes,
        do_lower_case=FLAGS.do_lower_case,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
        use_tpu=FLAGS.use_tpu,
        num_tpu_cores=FLAGS.num_tpu_cores,
        master=FLAGS.master
    )

    sentences_vector = bert_vector.convert_sentences_to_vectors(sentences=dataset.sentences)
    labels_vector = bert_vector.convert_sentences_to_vectors(sentences=labels_sentences)
    target = labels_id

    model = BertVectorClassify(
        clusters_init_value=labels_vector,
        units_list=[1024, 2048, 2048, 1024, 512]
    ).build()

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy(), ]
    )

    model.fit(x=sentences_vector, y=target)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
