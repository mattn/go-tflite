#!/bin/bash
# Seed the working directory with the example notebooks and models when they
# are not there yet, so they survive a host directory mounted over /notebooks.
for f in /usr/local/share/gophernotes-examples/*; do
  b=$(basename "$f")
  [ -e "/notebooks/$b" ] || cp "$f" "/notebooks/$b"
done
exec "$@"
