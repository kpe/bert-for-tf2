# coding=utf-8
#
# created by kpe on 15.Mar.2019 at 15:30
#

from __future__ import absolute_import, division, print_function


import unittest

import random
import numpy as np


import tensorflow as tf

from bert.attention import AttentionLayer


# tf.enable_v2_behavior()
# tf.enable_eager_execution()


class MaskFlatten(tf.keras.layers.Flatten):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, _, mask=None):
        return mask


class BertAttentionTest(unittest.TestCase):

    @staticmethod
    def data_generator(batch_size=32, max_len=10):             # ([batch_size, 10], [10])
        while True:
            data = np.zeros((batch_size, max_len))
            tag = np.zeros(batch_size, dtype='int32')
            for i in range(batch_size):
                datum_len = random.randint(1, max_len - 1)
                total = 0
                for j in range(datum_len):
                    data[i, j] = random.randint(1, 4)
                    total += data[i, j]
                tag[i] = total % 2
            yield data, tag

    def test_attention(self):
        max_seq_len = random.randint(5, 10)
        count = 0
        for data, tag in self.data_generator(4, max_seq_len):
            count += 1
            print(data, tag)
            if count > 2:
                break

        class AModel(tf.keras.models.Model):
            def __init__(self, **kwargs):
                super(AModel, self).__init__(**kwargs)
                self.embedding = tf.keras.layers.Embedding(input_dim=5, output_dim=3, mask_zero=True)
                self.attention = AttentionLayer(num_heads=5, size_per_head=3)
                self.timedist  = tf.keras.layers.TimeDistributed(MaskFlatten())
                self.bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))
                self.softmax = tf.keras.layers.Dense(units=2, activation="softmax")

            #def build(self, input_shape):
            #    super(AModel,self).build(input_shape)

            def call(self, inputs, training=None, mask=None):
                out = inputs
                out = self.embedding(out)
                out = self.attention(out)
                out = self.timedist(out)
                out = self.bigru(out)
                out = self.softmax(out)
                return out

        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=5, output_dim=3, mask_zero=True),
            AttentionLayer(num_heads=5, size_per_head=3),
            tf.keras.layers.TimeDistributed(MaskFlatten()),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8)),
            tf.keras.layers.Dense(units=2, activation="softmax")
        ])

        #model = AModel()
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        # model.build(input_shape=(None, max_seq_len))

        model.build()
        model.summary()

        model.fit_generator(
            generator=self.data_generator(64, max_seq_len),
            steps_per_epoch=100,
            epochs=10,
            validation_data=self.data_generator(8, max_seq_len),
            validation_steps=10,
            #callbacks=[
            #    keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
            #],
        )
