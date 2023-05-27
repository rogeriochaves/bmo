#!/usr/bin/env bash

set -eo pipefail

mkdir -p piper
cd piper

wget https://github.com/rhasspy/piper/releases/download/v0.0.2/piper_arm64.tar.gz
wget https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-ryan-medium.tar.gz

tar -xvf piper_arm64.tar.gz
tar -xvf voice-en-us-ryan-medium.tar.gz
