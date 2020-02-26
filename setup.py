#!/usr/bin/env python

#
# created by kpe on 22.10.2018 at 11:46
#

from setuptools import setup, find_packages, convert_path


def _version():
    ns = {}
    with open(convert_path("bert/version.py"), "r") as fh:
        exec(fh.read(), ns)
    return ns['__version__']


__version__ = _version()


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name="bert-for-tf2",
      version=__version__,
      url="https://github.com/kpe/bert-for-tf2/",
      description="A TensorFlow 2.0 Keras implementation of BERT.",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      keywords="tensorflow keras bert",
      license="MIT",

      author="kpe",
      author_email="kpe.git@gmailbox.org",
      packages=find_packages(exclude=["tests"]),
      package_data={"": ["*.txt", "*.rst"]},

      zip_safe=True,
      install_requires=install_requires,
      python_requires=">=3.5",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy"])
