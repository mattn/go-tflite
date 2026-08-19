#!/bin/bash
# Start Jupyter + gophernotes with a host directory mounted at /notebooks.
#
#   run.sh [directory]   directory to work in (default: current directory)
set -e
dir=$(cd "${1:-.}" && pwd)
exec docker run --rm -it -p 8888:8888 -v "$dir":/notebooks \
  ghcr.io/mattn/go-tflite/gophernotes
