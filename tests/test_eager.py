# coding=utf-8
#
# created by kpe on 25.Jul.2019 at 13:24
#

from __future__ import absolute_import, division, print_function


import numpy as np
import tensorflow as tf
from tensorflow import keras
import params_flow as pf

from bert import loader, BertModelLayer

from .test_common import AbstractBertTest


class LoaderTest(AbstractBertTest):

    def setUp(self) -> None:
        tf.reset_default_graph()
        tf.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    def test_coverage_improve(self):
        for act in ["relu", "gelu", "linear", None]:
            BertModelLayer.get_activation(act)
        try:
            BertModelLayer.get_activation("None")
        except ValueError:
            pass

    def test_eager_loading(self):
        print("Eager Execution:", tf.executing_eagerly())

        # a temporal mini bert model_dir
        model_dir = self.create_mini_bert_weights()

        bert_params = loader.params_from_pretrained_ckpt(model_dir)
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
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
                      run_eagerly=True)

        loader.load_stock_weights(bert, model_dir)

        model.summary()

    def test_concat(self):
        model_dir = self.create_mini_bert_weights()

        bert_params = loader.params_from_pretrained_ckpt(model_dir)
        bert_params.adapter_size = 32
        bert = BertModelLayer.from_params(bert_params, name="bert")

        max_seq_len = 4

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(max_seq_len,)),
            bert,
            keras.layers.TimeDistributed(keras.layers.Dense(bert_params.hidden_size)),
            keras.layers.TimeDistributed(keras.layers.LayerNormalization()),
            keras.layers.TimeDistributed(keras.layers.Activation("tanh")),

            pf.Concat([
                keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1)),  # GlobalMaxPooling1D
                keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1)),  # GlobalAvgPooling1
            ]),

            keras.layers.Dense(units=bert_params.hidden_size),
            keras.layers.Activation("tanh"),

            keras.layers.Dense(units=2)
        ])

        model.build(input_shape=(None, max_seq_len))
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
                      metrics=[keras.metrics.SparseCategoricalAccuracy()],
                      run_eagerly = True)

        loader.load_stock_weights(bert, model_dir)

        model.summary()
