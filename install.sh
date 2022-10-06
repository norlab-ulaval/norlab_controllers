#!/bin/bash

yes | pip uninstall norlabcontrollib
python setup.py bdist_wheel
pip install dist/norlabcontrollib-0.1.0-py3-none-any.whl 

