BERT for TensorFlow v2
======================

|Build Status| |Coverage Status| |Version Status| |Python Versions| |Downloads|

This repo contains a `TensorFlow 2.0`_ `Keras`_ implementation of `google-research/bert`_
with support for loading of the original `pre-trained weights`_,
and producing activations **numerically identical** to the one calculated by the original model.


The implementation is build from scratch using only basic tensorflow operations,
following the code in `google-research/bert/modeling.py`_
(but skipping dead code and applying some simplifications). It also utilizes `kpe/params-flow`_ to reduce
common Keras boilerplate code (related to passing model and layer configuration arguments).

`bert-for-tf2`_ should work with both `TensorFlow 2.0`_ and `TensorFlow 1.14`_ or newer.

NEWS
----

 - **25.Jul.2019** - there are now two colab notebooks under ``examples/`` showing how to
   fine-tune an IMDB Movie Reviews sentiment classifier from pre-trained BERT weights
   using an `adapter-BERT`_ model architecture on a GPU or TPU in Google Colab.

 - **28.Jun.2019** - v.0.3.0 supports `adapter-BERT`_ (`google-research/adapter-bert`_)
   for "Parameter-Efficient Transfer Learning for NLP", i.e. fine-tuning small overlay adapter
   layers over BERT's transformer encoders without changing the frozen BERT weights.



LICENSE
-------

MIT. See `License File <https://github.com/kpe/bert-for-tf2/blob/master/LICENSE.txt>`_.

Install
-------

``bert-for-tf2`` is on the Python Package Index (PyPI):

::

    pip install bert-for-tf2


Usage
-----

BERT in `bert-for-tf2` is implemented as a Keras layer. You could instantiate it like this:

.. code:: python

  from bert import BertModelLayer

  l_bert = BertModelLayer(BertModelLayer.Params(
    vocab_size               = 16000,        # embedding params
    use_token_type           = True,
    use_position_embeddings  = True,
    token_type_vocab_size    = 2,

    num_layers               = 12,           # transformer encoder params
    hidden_size              = 768,
    hidden_dropout           = 0.1,
    intermediate_size        = 4*768,
    intermediate_activation  = "gelu",

    adapter_size             = None,         # see arXiv:1902.00751

    name                     = "bert"        # any other Keras layer params
  ))

or by using the ``bert_config.json`` from a `pre-trained google model`_:

.. code:: python

  import tensorflow as tf
  from tensorflow import keras

  from bert import BertModelLayer
  from bert import params_from_pretrained_ckpt
  from bert import load_stock_weights

  model_dir = ".models/uncased_L-12_H-768_A-12"

  bert_params = params_from_pretrained_ckpt(model_dir)
  l_bert = BertModelLayer.from_params(bert_params, name="bert")


now you can use the BERT layer in your Keras model like this:

.. code:: python

  from tensorflow import keras

  max_seq_len = 128
  l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
  l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

  # using the default token_type/segment id 0
  output = l_bert(l_input_ids)                              # output: [batch_size, max_seq_len, hidden_size]
  model = keras.Model(inputs=l_input_ids, outputs=output)
  model.build(input_shape=(None, max_seq_len))

  # provide a custom token_type/segment id as a layer input
  output = l_bert([l_input_ids, l_token_type_ids])          # [batch_size, max_seq_len, hidden_size]
  model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
  model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

if you choose to use `adapter-BERT`_ by setting the `adapter_size` parameter,
you would also like to freeze all the original BERT layers by calling:

.. code:: python

  l_bert.apply_adapter_freeze()

and once the model has been build or compiled, the original pre-trained weights
can be loaded in the BERT layer:

.. code:: python

  from bert import load_stock_weights

  bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")
  load_stock_weights(l_bert, bert_ckpt_file)

**N.B.** see `tests/test_bert_activations.py`_ for a complete example.

Resources
---------

- `BERT`_ - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- `google-research/bert`_ - the original BERT implementation
- `kpe/params-flow`_ - A Keras coding style for reducing `Keras`_ boilerplate code in custom layers by utilizing `kpe/py-params`_


.. _`kpe/params-flow`: https://github.com/kpe/params-flow
.. _`kpe/py-params`: https://github.com/kpe/py-params
.. _`bert-for-tf2`: https://github.com/kpe/bert-for-tf2

.. _`Keras`: https://keras.io
.. _`pre-trained weights`: https://github.com/google-research/bert#pre-trained-models
.. _`google-research/bert`: https://github.com/google-research/bert
.. _`google-research/bert/modeling.py`: https://github.com/google-research/bert/blob/master/modeling.py
.. _`BERT`: https://arxiv.org/abs/1810.04805
.. _`pre-trained google model`: https://github.com/google-research/bert
.. _`tests/test_bert_activations.py`: https://github.com/kpe/bert-for-tf2/blob/master/tests/test_compare_activations.py
.. _`TensorFlow 2.0`: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf
.. _`TensorFlow 1.14`: https://www.tensorflow.org/versions/r1.14/api_docs/python/tf

.. _`google-research/adapter-bert`: https://github.com/google-research/adapter-bert/
.. _`adapter-BERT`: https://arxiv.org/abs/1902.00751

.. |Build Status| image:: https://travis-ci.org/kpe/bert-for-tf2.svg?branch=master
   :target: https://travis-ci.org/kpe/bert-for-tf2
.. |Coverage Status| image:: https://coveralls.io/repos/kpe/bert-for-tf2/badge.svg?branch=master
   :target: https://coveralls.io/r/kpe/bert-for-tf2?branch=master
.. |Version Status| image:: https://badge.fury.io/py/bert-for-tf2.svg
   :target: https://badge.fury.io/py/bert-for-tf2
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/bert-for-tf2.svg
.. |Downloads| image:: https://img.shields.io/pypi/dm/bert-for-tf2.svg
.. |Twitter| image:: https://img.shields.io/twitter/follow/siddhadev?logo=twitter&label=&style=
   :target: https://twitter.com/intent/user?screen_name=siddhadev