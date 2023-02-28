#!/bin/bash

yes | pip3 uninstall norlabcontrollib
python3 setup.py bdist_wheel
pip3 install dist/norlabcontrollib-0.1.0-py3-none-any.whl 

