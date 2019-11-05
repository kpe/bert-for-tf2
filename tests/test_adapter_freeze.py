# coding=utf-8
#
# created by kpe on 09.08.2019 at 10:38 PM
#

from __future__ import division, absolute_import, print_function


import unittest

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import bert

from .test_common import AbstractBertTest, MiniBertFactory


class AdapterFreezeTest(AbstractBertTest):

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    def test_adapter_freezing(self):
        bert_params = bert.BertModelLayer.Params(hidden_size=32,
                                                 vocab_size=67,
                                                 max_position_embeddings=64,
                                                 num_layers=1,
                                                 num_heads=1,
                                                 intermediate_size=4,
                                                 use_token_type=False)

        def to_model(bert_params):
            l_bert = bert.BertModelLayer.from_params(bert_params)

            token_ids = keras.layers.Input(shape=(21,))
            seq_out = l_bert(token_ids)
            model = keras.Model(inputs=[token_ids], outputs=seq_out)

            model.build(input_shape=(None, 21))
            l_bert.apply_adapter_freeze()

            return model

        model = to_model(bert_params)
        model.summary()
        print("trainable wegihts:", len(model.trainable_weights))
        self.assertEqual(20, len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)

        bert_params.adapter_size = 16

        model = to_model(bert_params)
        model.summary()
        print("trainable weights:", len(model.trainable_weights))
        self.assertEqual(14, len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)

    def test_bert_freeze(self):
        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = 24
        input_str_batch    = ["hello, bert!", "how are you doing!"]

        input_ids, token_type_ids = self.prepare_input_batch(input_str_batch, tokenizer, max_seq_len)

        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        bert_params.adapter_size = 4
        l_bert = bert.BertModelLayer.from_params(bert_params)

        model = keras.models.Sequential([
            l_bert,
        ])

        model.build(input_shape=(None, max_seq_len))

        model.summary()
        l_bert.apply_adapter_freeze()
        model.summary()

        bert.load_stock_weights(l_bert, bert_ckpt_file)
        #l_bert.embeddings_layer.trainable = False

        model.summary()

        orig_weight_values = []
        for weight in l_bert.weights:
            orig_weight_values.append(weight.numpy())

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error,
                      run_eagerly=True)

        trainable_count = len(l_bert.trainable_weights)

        orig_pred = model.predict(input_ids)
        model.fit(x=input_ids, y=np.zeros_like(orig_pred),
          batch_size=2,
          epochs=4)

        trained_count = 0
        for ndx, weight in enumerate(l_bert.weights):
            weight_equal = np.array_equal(weight.numpy(), orig_weight_values[ndx])
            print("{}: {}".format(weight_equal, weight.name))
            if not weight_equal:
                trained_count += 1

        print("  trained weights:", trained_count)
        print("trainable weights:", trainable_count)
        self.assertEqual(trained_count, trainable_count)

        model.summary()

    def test_adapter_albert_freeze(self):
        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        # for tokenizer only
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = 28
        input_str_batch    = ["hello, albert!", "how are you doing!"]
        input_ids, token_type_ids = self.prepare_input_batch(input_str_batch, tokenizer, max_seq_len,
                                                             extra_token_count=3)

        bert_params = bert.BertModelLayer.Params(
            attention_dropout=0.1,
            hidden_act="gelu",
            hidden_dropout=0.1,
            hidden_size=8,
            initializer_range=0.02,
            intermediate_size=32,
            max_position_embeddings=32,
            num_heads=2,
            num_layers=2,
            token_type_vocab_size=2,
            vocab_size=len(tokenizer.vocab),

            adapter_size=2,

            embedding_size=4,
            extra_tokens_vocab_size=3,
            shared_layer=True,
        )
        l_bert = bert.BertModelLayer.from_params(bert_params)

        model = keras.models.Sequential([
            l_bert,
        ])

        model.build(input_shape=(None, max_seq_len))

        model.summary()
        l_bert.apply_adapter_freeze()
        model.summary()

        orig_weight_values = []
        for weight in l_bert.weights:
            orig_weight_values.append(weight.numpy())

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error,
                      run_eagerly=True)

        trainable_count = len(l_bert.trainable_weights)

        orig_pred = model.predict(input_ids)
        model.fit(x=input_ids, y=np.zeros_like(orig_pred),
          batch_size=2,
          epochs=4)

        trained_count = 0
        for ndx, weight in enumerate(l_bert.weights):
            weight_equal = np.array_equal(weight.numpy(), orig_weight_values[ndx])
            print("trained:[{}]: {}".format(not weight_equal, weight.name))
            if not weight_equal:
                trained_count += 1

        print("  trained weights:", trained_count)
        print("trainable weights:", trainable_count)
        self.assertEqual(trained_count, trainable_count)

        model.summary()






