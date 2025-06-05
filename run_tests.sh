#!/bin/sh
# Simple helper for CI/devs: install minimal test requirements and run pytest
set -e
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
pytest -q
