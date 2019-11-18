# coding=utf-8
#
# created by kpe on 02.Sep.2019 at 11:57
#

from __future__ import absolute_import, division, print_function

import unittest

import os
import re
import tempfile

import bert

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .test_common import AbstractBertTest, MiniBertFactory

#tf.enable_eager_execution()
#tf.disable_eager_execution()


class TestExtendSegmentVocab(AbstractBertTest):

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    def test_extend_pretrained_segments(self):

        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        ckpt_dir = os.path.dirname(save_path)
        bert_params = bert.params_from_pretrained_ckpt(ckpt_dir)

        self.assertEqual(bert_params.token_type_vocab_size, 2)
        bert_params.token_type_vocab_size = 4

        l_bert = bert.BertModelLayer.from_params(bert_params)

        # we dummy call the layer once in order to instantiate the weights
        l_bert([np.array([[1, 1, 0]]),
                np.array([[1, 0, 0]])])#, mask=[[True, True, False]])

        #
        # - load the weights from a pre-trained model,
        # - expect a mismatch for the token_type embeddings
        # - use the segment/token type id=0 embedding for the missing token types
        #
        mismatched = bert.load_stock_weights(l_bert, save_path)

        self.assertEqual(1, len(mismatched), "token_type embeddings should have mismatched shape")

        for weight, value in mismatched:
            if re.match("(.*)embeddings/token_type_embeddings/embeddings:0", weight.name):
                seg0_emb = value[:1, :]
                new_segment_embeddings = np.repeat(seg0_emb, (weight.shape[0]-value.shape[0]), axis=0)
                new_value = np.concatenate([value, new_segment_embeddings], axis=0)
                keras.backend.batch_set_value([(weight, new_value)])

        tte = l_bert.embeddings_layer.token_type_embeddings_layer.weights[0]

        if not tf.executing_eagerly():
            with tf.keras.backend.get_session() as sess:
                tte, = sess.run((tte, ))

        self.assertTrue(np.allclose(seg0_emb, tte[0], 1e-6))
        self.assertFalse(np.allclose(seg0_emb, tte[1], 1e-6))
        self.assertTrue(np.allclose(seg0_emb, tte[2], 1e-6))
        self.assertTrue(np.allclose(seg0_emb, tte[3], 1e-6))

        bert_params.token_type_vocab_size = 4
        print("token_type_vocab_size", bert_params.token_type_vocab_size)
        print(l_bert.embeddings_layer.trainable_weights[1])


