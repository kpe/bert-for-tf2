# coding=utf-8
#
# created by kpe on 25.Jul.2019 at 12:23
#

from __future__ import absolute_import, division, print_function



import unittest
import math

import tensorflow as tf
from tensorflow.python import keras

from bert import BertModelLayer, loader
from bert.loader import map_from_stock_variale_name, map_to_stock_variable_name, load_stock_weights
from bert.loader import StockBertConfig, map_stock_config_to_params
from bert.tokenization import FullTokenizer

#tf.enable_eager_execution()
#tf.disable_eager_execution()


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


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

        # we need to freeze before build/compile - otherwise keras counts the params twice
        if bert_params.adapter_size is not None:
            freeze_bert_layers(bert)

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        load_stock_weights(bert, self.bert_ckpt_file)

        model.summary()


