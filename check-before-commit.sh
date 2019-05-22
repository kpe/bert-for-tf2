#!/bin/bash

PEP8_IGNORE=E221,E501,W504,W391,E241

pep8 --ignore=${PEP8_IGNORE} --exclude=tests,.venv -r --show-source tests bert

coverage run --source=bert /home/kpe/proj/local/params-flow.kpe/.venv/bin/nosetests --with-doctest tests/
coverage report --show-missing --fail-under=100
