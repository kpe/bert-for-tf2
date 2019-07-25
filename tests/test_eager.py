# coding=utf-8
#
# created by kpe on 25.Jul.2019 at 13:24
#

from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow import keras

from bert import loader, BertModelLayer

from .test_common import AbstractBertTest

tf.enable_eager_execution()

print("Eager Execution:", tf.executing_eagerly())


class LoaderTest(AbstractBertTest):

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
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        loader.load_stock_weights(bert, model_dir)

        model.summary()
