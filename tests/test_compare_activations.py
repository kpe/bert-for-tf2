# coding=utf-8
#
# created by kpe on 23.May.2019 at 17:10
#

from __future__ import absolute_import, division, print_function

import os
import string
import unittest
import tempfile

import tensorflow as tf
import numpy as np

from tensorflow import keras

from bert import bert_tokenization

from .test_common import AbstractBertTest, MiniBertFactory


class CompareBertActivationsTest(AbstractBertTest):

    def setUp(self):
        tf.compat.v1.reset_default_graph()
        keras.backend.clear_session()
        tf.compat.v1.disable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    @staticmethod
    def load_stock_model(model_dir, max_seq_len):
        from tests.ext.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

        tf.compat.v1.reset_default_graph()  # to scope naming for checkpoint loading (if executed more than once)

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        pl_input_ids      = tf.compat.v1.placeholder(tf.int32, shape=(1, max_seq_len))
        pl_mask           = tf.compat.v1.placeholder(tf.int32, shape=(1, max_seq_len))
        pl_token_type_ids = tf.compat.v1.placeholder(tf.int32, shape=(1, max_seq_len))

        bert_config = BertConfig.from_json_file(bert_config_file)

        s_model = BertModel(config=bert_config,
                            is_training=False,
                            input_ids=pl_input_ids,
                            input_mask=pl_mask,
                            token_type_ids=pl_token_type_ids,
                            use_one_hot_embeddings=False)

        tvars = tf.compat.v1.trainable_variables()
        (assignment_map, initialized_var_names) = get_assignment_map_from_checkpoint(tvars, bert_ckpt_file)
        tf.compat.v1.train.init_from_checkpoint(bert_ckpt_file, assignment_map)

        return s_model, pl_input_ids, pl_token_type_ids, pl_mask

    @staticmethod
    def predict_on_stock_model(model_dir, input_ids, input_mask, token_type_ids):
        max_seq_len = input_ids.shape[-1]
        (s_model,
         pl_input_ids, pl_token_type_ids, pl_mask) = CompareBertActivationsTest.load_stock_model(model_dir, max_seq_len)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            s_res = sess.run(
                s_model.get_sequence_output(),
                feed_dict={pl_input_ids:      input_ids,
                           pl_token_type_ids: token_type_ids,
                           pl_mask:           input_mask,
                           })
        return s_res

    @staticmethod
    def load_keras_model(model_dir, max_seq_len):
        from tensorflow.python import keras
        from bert import BertModelLayer
        from bert.loader import StockBertConfig, load_stock_weights, params_from_pretrained_ckpt

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        l_bert = BertModelLayer.from_params(params_from_pretrained_ckpt(model_dir))

        l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")

        output = l_bert([l_input_ids, l_token_type_ids])

        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        load_stock_weights(l_bert, bert_ckpt_file)
        return model

    @staticmethod
    def predict_on_keras_model(model_dir, input_ids, input_mask, token_type_ids):
        max_seq_len = input_ids.shape[-1]

        model = CompareBertActivationsTest.load_keras_model(model_dir, max_seq_len)

        k_res = model.predict([input_ids, token_type_ids])
        return k_res

    def test_compare(self):

        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert_tokenization.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = 16
        input_str    = "hello, bert!"
        input_tokens = tokenizer.tokenize(input_str)
        input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
        input_ids    = tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids      = input_ids             + [0]*(max_seq_len - len(input_tokens))
        input_mask     = [0]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens)) # FIXME: input_mask broken - chane to [1]*
        token_type_ids = [0]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens))

        input_ids      = np.array([input_ids], dtype=np.int32)
        input_mask     = np.array([input_mask], dtype=np.int32)
        token_type_ids = np.array([token_type_ids], dtype=np.int32)

        print("   tokens:", input_tokens)
        print("input_ids:{}/{}:{}".format(len(input_tokens), max_seq_len, input_ids), input_ids.shape, token_type_ids)

        bert_1_seq_out = CompareBertActivationsTest.predict_on_stock_model(model_dir, input_ids, input_mask, token_type_ids)
        bert_2_seq_out = CompareBertActivationsTest.predict_on_keras_model(model_dir, input_ids, input_mask, token_type_ids)

        np.set_printoptions(precision=9, threshold=20, linewidth=200, sign="+", floatmode="fixed")

        print("stock bert res", bert_1_seq_out.shape)
        print("keras bert res", bert_2_seq_out.shape)

        print("stock bert res:\n {}".format(bert_1_seq_out[0, :2, :10]), bert_1_seq_out.dtype)
        print("keras bert_res:\n {}".format(bert_2_seq_out[0, :2, :10]), bert_2_seq_out.dtype)

        abs_diff = np.abs(bert_1_seq_out - bert_2_seq_out).flatten()
        print("abs diff:", np.max(abs_diff), np.argmax(abs_diff))
        self.assertTrue(np.allclose(bert_1_seq_out, bert_2_seq_out, atol=1e-6))

    def test_finetune(self):


        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert_tokenization.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = 24
        input_str_batch    = ["hello, bert!", "how are you doing!"]

        input_ids_batch    = []
        token_type_ids_batch = []
        for input_str in input_str_batch:
            input_tokens = tokenizer.tokenize(input_str)
            input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]

            print("input_tokens len:", len(input_tokens))

            input_ids      = tokenizer.convert_tokens_to_ids(input_tokens)
            input_ids      = input_ids             + [0]*(max_seq_len - len(input_tokens))
            token_type_ids = [0]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens))

            input_ids_batch.append(input_ids)
            token_type_ids_batch.append(token_type_ids)

        input_ids      = np.array(input_ids_batch, dtype=np.int32)
        token_type_ids = np.array(token_type_ids_batch, dtype=np.int32)

        print("   tokens:", input_tokens)
        print("input_ids:{}/{}:{}".format(len(input_tokens), max_seq_len, input_ids), input_ids.shape, token_type_ids)

        model = CompareBertActivationsTest.load_keras_model(model_dir, max_seq_len)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error)

        pres = model.predict([input_ids, token_type_ids])  # just for fetching the shape of the output
        print("pres:", pres.shape)

        model.fit(x=(input_ids, token_type_ids),
                  y=np.zeros_like(pres),
                  batch_size=2,
                  epochs=2)
