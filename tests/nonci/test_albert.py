# coding=utf-8
#
# created by kpe on 10.Oct.2019 at 16:26
#

from __future__ import absolute_import, division, print_function

import unittest

import tensorflow as tf
from tensorflow import keras

import bert


class TestAlbertLoadWeights(unittest.TestCase):
    bert_ckpt_dir = ".models/albert_base_zh/"
    bert_ckpt_file = bert_ckpt_dir + "albert_model.ckpt"
    bert_config_file = bert_ckpt_dir + "albert_config_base.json"

    def test_load_pretrained(self):
        print("Eager Execution:", tf.executing_eagerly())

        bert_params = bert.loader.params_from_pretrained_ckpt(self.bert_ckpt_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(128,)),
            l_bert,
            keras.layers.Lambda(lambda x: x[:, 0, :]),
            keras.layers.Dense(2)
        ])

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        for weight in l_bert.weights:
            print(weight.name)

        bert.loader.load_stock_weights(l_bert, self.bert_ckpt_file)

        model.summary()
