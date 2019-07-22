# coding=utf-8
#
# created by kpe on 22.Jul.2019 at 16:02
#

from __future__ import absolute_import, division, print_function

import os
import re
import math
import datetime

import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import keras

tf.compat.v1.enable_eager_execution()

#
# https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
#


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True)

  train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      "aclImdb", "test"))

  return train_df, test_df


from tqdm import tqdm

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization import FullTokenizer


class MovieReviewData:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "polarity"

    def __init__(self, tokenizer: FullTokenizer, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test = download_and_load_datasets()
        if sample_size is not None:
            train, test = map(lambda df: df.sample(sample_size), [train, test])
        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad, [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[MovieReviewData.DATA_COLUMN], row[MovieReviewData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return x, y

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)


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


class TestOnMovieReviews(unittest.TestCase):
    bert_ckpt_dir = ".models/uncased_L-12_H-768_A-12/"
    bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
    bert_config_file = bert_ckpt_dir + "bert_config.json"

    def test_movie_reviews(self):
        tokenizer = FullTokenizer(vocab_file=os.path.join(self.bert_ckpt_dir, "vocab.txt"))
        data = MovieReviewData(tokenizer, sample_size=5000, max_seq_len=128)  # for 256 or 512 reduce the batch_size

        print("sample_size", data.sample_size)
        print("max_seq_len", data.max_seq_len)

        print("             train_x", data.train_x.shape)
        print(" train_x_token_types", data.train_x_token_types.shape)

        max_seq_len = data.max_seq_len

        adapter_size = 64  # see - arXiv:1902.00751

        # create the bert layer
        with tf.io.gfile.GFile(self.bert_config_file, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = adapter_size
            bert = BertModelLayer.from_params(bert_params, name="bert")

        input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
        output         = bert([input_ids, token_type_ids])

        print("bert shape", output.shape)
        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)
        logits = keras.layers.Dense(units=768, activation="relu")(cls_out)
        logits = keras.layers.Dropout(0.5)(logits)
        logits = keras.layers.Dense(units=2, activation="softmax")(logits)

        model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        # load the pre-trained model weights
        load_stock_weights(bert, self.bert_ckpt_file)

        # freeze weights if adapter-BERT is used
        if adapter_size is not None:
            freeze_bert_layers(bert)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        model.summary()

        log_dir = ".log/movie_reviews/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        print("            train_x", np.array(data.train_x).shape)
        print("train_x_token_types", np.array(data.train_x_token_types).shape)

        total_epoch_count = 100
        model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
                  batch_size=32,
                  shuffle=True,
                  validation_split=0.1,
                  epochs=total_epoch_count,
                  callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-4,
                                                            end_learn_rate=1e-7,
                                                            warmup_epoch_count=20,
                                                            total_epoch_count=total_epoch_count),
                             keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                             tensorboard_callback])

        print("Evaluating...")
        _, train_acc = model.evaluate((data.train_x, data.train_x_token_types), data.train_y)
        _, test_acc = model.evaluate((data.test_x, data.test_x_token_types), data.test_y)
        print("train acc:", train_acc)
        print(" test acc:", test_acc)
