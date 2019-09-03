# coding=utf-8
#
# created by kpe on 02.Sep.2019 at 23:57
#

from __future__ import absolute_import, division, print_function


import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import bert

from .test_common import MiniBertFactory, AbstractBertTest


class TestAdapterFineTuning(AbstractBertTest):
    """
    Demonstrates a fine tuning workflow using adapte-BERT with
    storing the fine tuned and frozen pre-trained weights in
    separate checkpoint files.
    """

    def setUp(self) -> None:
        tf.reset_default_graph()
        tf.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

        # build a dummy bert
        self.ckpt_path = MiniBertFactory.create_mini_bert_weights()
        self.ckpt_dir = os.path.dirname(self.ckpt_path)
        self.tokenizer = bert.FullTokenizer(vocab_file=os.path.join(self.ckpt_dir, "vocab.txt"), do_lower_case=True)

    def test_coverage_improve(self):
        bert_params = bert.params_from_pretrained_ckpt(self.ckpt_dir)
        model, l_bert = self.build_model(bert_params, 1)
        for weight in model.weights:
            try:
                name = bert.loader.map_to_stock_variable_name(weight.name, weight.name.split("/")[0])
                stock_name = bert.loader.map_from_stock_variale_name(name)
                self.assertEqual(name, stock_name)
            except:
                print(weight.name)


    @staticmethod
    def build_model(bert_params, max_seq_len):
        # enable adapter-BERT
        bert_params.adapter_size = 2
        l_bert = bert.BertModelLayer.from_params(bert_params)
        model = keras.models.Sequential([
            l_bert,
            keras.layers.Lambda(lambda seq: seq[:, 0, :]),
            keras.layers.Dense(3)
        ])
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

        # build for a given max_seq_len
        model.build(input_shape=(None, max_seq_len))
        return model, l_bert

    def test_finetuning_workflow(self):
        # create a BERT layer with config from the checkpoint
        bert_params = bert.params_from_pretrained_ckpt(self.ckpt_dir)

        max_seq_len = 12

        model, l_bert = self.build_model(bert_params, max_seq_len=max_seq_len)
        model.summary()

        # freeze non-adapter weights
        l_bert.apply_adapter_freeze()
        model.summary()

        # load the BERT weights from the pre-trained model
        bert.load_stock_weights(l_bert, self.ckpt_path)

        # prepare the data
        inputs, targets = ["hello world", "goodbye"], [1, 2]
        tokens = [self.tokenizer.tokenize(toks) for toks in inputs]
        tokens = [self.tokenizer.convert_tokens_to_ids(toks) for toks in tokens]
        tokens = [toks + [0]*(max_seq_len - len(toks)) for toks in tokens]
        x = np.array(tokens)
        y = np.array(targets)

        # fine tune
        model.fit(x, y, epochs=3)

        # preserve the logits for comparison before and after restoring the fine-tuned model
        logits = model.predict(x)

        # now store the adapter weights only

        # old fashion - using saver
        #  finetuned_weights = {w.name: w.value() for w in model.trainable_weights}
        #  saver = tf.compat.v1.train.Saver(finetuned_weights)
        #  fine_path = saver.save(tf.compat.v1.keras.backend.get_session(), fine_ckpt)

        fine_ckpt = os.path.join(self.ckpt_dir, "fine-tuned.ckpt")
        finetuned_weights = {w.name: w for w in model.trainable_weights}
        checkpoint = tf.train.Checkpoint(**finetuned_weights)
        fine_path = checkpoint.save(file_prefix=fine_ckpt)
        print("fine tuned ckpt:", fine_path)

        # build new model
        tf.compat.v1.keras.backend.clear_session()
        model, l_bert = self.build_model(bert_params, max_seq_len=max_seq_len)
        l_bert.apply_adapter_freeze()

        # load the BERT weights from the pre-trained checkpoint
        bert.load_stock_weights(l_bert, self.ckpt_path)

        # load the fine tuned classifier model weights
        finetuned_weights = {w.name: w for w in model.trainable_weights}
        checkpoint = tf.train.Checkpoint(**finetuned_weights)
        load_status = checkpoint.restore(fine_path)
        load_status.assert_consumed().run_restore_ops()

        logits_restored = model.predict(x)

        # check the predictions of the restored model
        self.assertTrue(np.allclose(logits_restored, logits, 1e-6))
