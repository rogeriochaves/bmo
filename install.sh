#!/usr/bin/env bash

set -eo pipefail

if [ "$(uname)" == "Darwin" ]; then
  brew install libunistring sdl2
else
  sudo apt-get install -y libsdl2-dev espeak-ng
fi

pip install -r requirements.txt
