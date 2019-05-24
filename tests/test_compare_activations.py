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

tf.compat.v1.disable_eager_execution()

class MiniBertFactory:

    @staticmethod
    def create_mini_bert_weights(model_dir):
        from bert.loader import StockBertConfig

        bert_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        bert_config = StockBertConfig(
            attention_probs_dropout_prob = 0.1,
            hidden_act                   = "gelu",
            hidden_dropout_prob          = 0.1,
            hidden_size                  = 8,
            initializer_range            = 0.02,
            intermediate_size            = 32,
            max_position_embeddings      = 16,
            num_attention_heads          = 2,
            num_hidden_layers            = 2,
            type_vocab_size              = 2,
            vocab_size                   = len(string.ascii_lowercase)*2 + len(bert_tokens)
        )

        print("creating mini BERT at:", model_dir)

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_vocab_file  = os.path.join(model_dir, "vocab.txt")

        with open(bert_config_file, "w") as f:
            f.write(bert_config.to_json_string())
        with open(bert_vocab_file, "w") as f:
            f.write("\n".join(list(string.ascii_lowercase) + bert_tokens))
            f.write("\n".join(["##"+tok for tok in list(string.ascii_lowercase)]))

        with tf.Graph().as_default():
            _ = MiniBertFactory.create_stock_bert_graph(bert_config_file, 16)
            saver = tf.compat.v1.train.Saver(max_to_keep=1, save_relative_paths=True)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                ckpt_path = os.path.join(model_dir, "bert_model.ckpt")
                save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
                print("saving to:", save_path)

        return save_path

    @staticmethod
    def create_stock_bert_graph(bert_config_file, max_seq_len):
        from tests.ext.modeling import BertModel, BertConfig

        tf_placeholder = tf.compat.v1.placeholder

        pl_input_ids      = tf_placeholder(tf.int32, shape=(1, max_seq_len))
        pl_mask           = tf_placeholder(tf.int32, shape=(1, max_seq_len))
        pl_token_type_ids = tf_placeholder(tf.int32, shape=(1, max_seq_len))

        bert_config = BertConfig.from_json_file(bert_config_file)
        s_model = BertModel(config=bert_config,
                            is_training=False,
                            input_ids=pl_input_ids,
                            input_mask=pl_mask,
                            token_type_ids=pl_token_type_ids,
                            use_one_hot_embeddings=False)

        return s_model, pl_input_ids, pl_mask, pl_token_type_ids


class CompareBertActivationsTest(unittest.TestCase):

    @staticmethod
    def create_mini_bert_weights():
        model_dir = tempfile.TemporaryDirectory().name
        # model_dir = "/tmp/mini_bert/";
        os.makedirs(model_dir, exist_ok=True)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        print("mini_bert save_path", save_path)
        print("\n\t".join([""] + os.listdir(model_dir)))
        return model_dir

    @staticmethod
    def predict_on_stock_model(model_dir, input_ids, input_mask, token_type_ids):
        from tests.ext.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        tf_placeholder = tf.compat.v1.placeholder

        max_seq_len       = input_ids.shape[-1]
        pl_input_ids      = tf_placeholder(tf.int32, shape=(1, max_seq_len))
        pl_mask           = tf_placeholder(tf.int32, shape=(1, max_seq_len))
        pl_token_type_ids = tf_placeholder(tf.int32, shape=(1, max_seq_len))

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
    def predict_on_keras_model(model_dir, input_ids, input_mask, token_type_ids):
        from tensorflow.python import keras
        from bert import BertModelLayer
        from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        max_seq_len = input_ids.shape[-1]

        bert = None
        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert = BertModelLayer.from_params(map_stock_config_to_params(bc),
                                              name="bert")

        l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
        output = bert([input_ids, token_type_ids])

        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])
        load_stock_weights(bert, bert_ckpt_file)

        k_res = model.predict([input_ids, token_type_ids])
        return k_res

    def test_compare(self):
        from tests.ext.tokenization import FullTokenizer

        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = 16
        input_str    = "hello, bert!"
        input_tokens = tokenizer.tokenize(input_str)
        input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
        input_ids    = tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids      = input_ids             + [0]*(max_seq_len - len(input_tokens))
        input_mask     = [1]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens))
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
        self.assertTrue(np.allclose(bert_1_seq_out, bert_2_seq_out, atol=1e-8))
