#!/usr/bin/env bash

set -e

if [[ -z "${VIRTUAL_ENV}" ]]; then
    >&2 echo "No active virtual environment found."
    exit 1
fi

pip install --upgrade pip
pip install --upgrade --force-reinstall -e .
pip install pytest==9.0.1 black==25.11.0
