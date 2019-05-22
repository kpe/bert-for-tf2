#!/usr/bin/env python

#
# created by kpe on 22.10.2018 at 11:46
#


from setuptools import setup

import bert


with open("README.rst", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name='bert-for-tf2',
      version='0.0.2',
      description="A TensorFlow 2.0 Keras implementation of BERT.",
      url="https://github.com/kpe/bert-for-tf2/",
      author="kpe",
      author_email="kpe.git@gmailbox.org",
      license="MIT",
      keywords="tensorflow keras bert",
      packages=["bert"],
      package_data={"bert": ["tests/*.py", "requirements.txt"]},
      long_description=long_description,
      long_description_content_type="text/x-rst",
      zip_safe=False,
      install_requires=install_requires,
      python_requires=">=3.4",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy"])
