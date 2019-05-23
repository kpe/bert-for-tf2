#!/bin/bash

PEP8_IGNORE=E221,E501,W504,W391,E241

pep8 --ignore=${PEP8_IGNORE} --exclude=tests,.venv -r --show-source tests bert

coverage run --source=bert $(which nosetests) -v --with-doctest tests/ --exclude-dir tests/nonci/
coverage report --show-missing --fail-under=50
