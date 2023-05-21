#!/usr/bin/env bash

set -eo pipefail

sudo apt-get install libatlas-base-dev

pip install -r requirements.txt
