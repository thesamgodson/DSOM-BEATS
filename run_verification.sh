#!/bin/bash
set -e
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install jupyter nbformat
python examples/usage_tutorial.py
