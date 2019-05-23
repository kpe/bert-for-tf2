BERT for TensorFlow v2
======================

|Build Status| |Coverage Status| |Version Status| |Python Versions|

This repo contains a TensorFlow v2 Keras implementation of `google-research/bert`_
with support for loading the original `pre-trained weights`_,
and producing activations numerically identical to the one you get from the original model.


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

TBD


Resources
---------

- `BERT`_ - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- `google-research/bert`_ - the original BERT implementation
- `kpe/params-flow`_ - utilities for reducing keras boilerplate code in custom layers

.. _`pre-trained weights`: https://github.com/google-research/bert#pre-trained-models
.. _`google-research/bert`: https://github.com/google-research/bert
.. _`BERT`: https://arxiv.org/abs/1810.04805
.. _`kpe/params-flow`: https://github.com/kpe/params-flow

.. |Build Status| image:: https://travis-ci.org/kpe/bert-for-tf2.svg?branch=master
   :target: https://travis-ci.org/kpe/bert-for-tf2
.. |Coverage Status| image:: https://coveralls.io/repos/kpe/bert-for-tf2/badge.svg?branch=master
   :target: https://coveralls.io/r/kpe/bert-for-tf2?branch=master
.. |Version Status| image:: https://badge.fury.io/py/bert-for-tf2.svg
   :target: https://badge.fury.io/py/bert-for-tf2
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/bert-for-tf2.svg

