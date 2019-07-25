# coding=utf-8
#
# created by kpe on 25.Jul.2019 at 12:23
#

from __future__ import absolute_import, division, print_function



import unittest

import tensorflow as tf
from tensorflow.python import keras

from bert import BertModelLayer, loader
from bert.loader import map_from_stock_variale_name, map_to_stock_variable_name, load_stock_weights
from bert.loader import StockBertConfig, map_stock_config_to_params
from bert.tokenization import FullTokenizer

#tf.enable_eager_execution()
#tf.disable_eager_execution()


class TestWeightsLoading(unittest.TestCase):
    bert_ckpt_dir = ".models/uncased_L-12_H-768_A-12/"
    bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
    bert_config_file = bert_ckpt_dir + "bert_config.json"

    def test_load_pretrained(self):
        print("Eager Execution:", tf.executing_eagerly())

        bert_params = loader.params_from_pretrained_ckpt(self.bert_ckpt_dir)
        bert_params.adapter_size = 32
        bert = BertModelLayer.from_params(bert_params, name="bert")

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(128,)),
            bert,
            keras.layers.Lambda(lambda x: x[:, 0, :]),
            keras.layers.Dense(2)
        ])

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        load_stock_weights(bert, self.bert_ckpt_file)

        model.summary()


