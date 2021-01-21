# coding=utf-8
#
# created by kpe on 21.Mar.2019 at 13:30
#

from __future__ import absolute_import, division, print_function

import random

import unittest

import numpy as np
import tensorflow as tf

from tensorflow import keras


from bert.transformer import TransformerEncoderLayer


class MaskFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, _, mask=None):
        return mask


def parity_ds_generator(batch_size=32, max_len=10, max_int=4, modulus=2):
    """
    Generates a parity calculation dataset (seq -> sum(seq) mod 2),
    where seq is a sequence of length less than max_len
    of integers in [1..max_int).
    """
    while True:
        data = np.zeros((batch_size, max_len))
        tag = np.zeros(batch_size, dtype='int32')
        for i in range(batch_size):
            datum_len = random.randint(1, max_len - 1)
            total = 0
            for j in range(datum_len):
                data[i, j] = random.randint(1, max_int)
                total += data[i, j]
            tag[i] = total % modulus
        yield data, tag                  # ([batch_size, max_len], [max_len])


class TransformerTest(unittest.TestCase):

    def test_simple(self):
        max_seq_len = 10
        model = keras.Sequential([
            keras.layers.Embedding(input_dim=5, output_dim=15, mask_zero=True),      # [B, 10, 12]
            TransformerEncoderLayer(
                hidden_size=15,
                num_heads=5,
                num_layers=2,
                intermediate_size=8,
                hidden_dropout=0.1),                                        # [B, 10, 6]
            keras.layers.TimeDistributed(
                keras.layers.Dense(units=2, activation="softmax")),         # [B, 10, 2]
            keras.layers.Lambda(lambda x: x[:, -0, ...])                    # [B, 2]
            ])

        model.build(input_shape=(None, max_seq_len))

        model.compile(optimizer=keras.optimizers.Adam(lr=0.003),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy]
                      )
        model.summary(line_length=120)

        model.fit_generator(generator=parity_ds_generator(64, max_seq_len),
                            steps_per_epoch=100,
                            epochs=20,
                            validation_data=parity_ds_generator(12, -4+max_seq_len),
                            validation_steps=10,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
                            ],
                            )




