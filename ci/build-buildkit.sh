#!/bin/bash
# Build the go-tflite buildkit tarball.
#
# The buildkit is a tarball that, when extracted into /usr/local, provides
# everything needed for `go build` of go-tflite to succeed:
#
#   include/tensorflow/lite/...      TensorFlow Lite headers (only the ones
#                                    actually included, computed via gcc -M)
#   lib/libtensorflowlite_c.so       TensorFlow Lite C API shared library
#   lib/libtensorflowlite-delegate_xnnpack.so
#                                    XNNPACK delegate shared library
#   lib/libXNNPACK.so                XNNPACK itself
#
# Run from the go-tflite repository root.
#
# Environment variables:
#   TENSORFLOW_VERSION  git tag/branch of tensorflow to build (default: v2.17.1)
#   TENSORFLOW_SRC      where to clone/find the tensorflow source
#   OUT_DIR             where to place the resulting tarball (default: ./dist)
#   BUILDKIT_SUFFIX     suffix of the tarball name, typically a go-tflite
#                       release tag (default: today's date as YYYYMMDD)
set -euo pipefail

TENSORFLOW_VERSION=${TENSORFLOW_VERSION:-v2.17.1}
TENSORFLOW_SRC=${TENSORFLOW_SRC:-$HOME/tensorflow_src}
OUT_DIR=${OUT_DIR:-$PWD/dist}
GO_TFLITE_ROOT=$PWD

if [ ! -d "$TENSORFLOW_SRC" ]; then
  git clone --depth 1 --branch "$TENSORFLOW_VERSION" \
    https://github.com/tensorflow/tensorflow "$TENSORFLOW_SRC"
fi

cd "$TENSORFLOW_SRC"

# Non-interactive configure: CPU only, no Android, plain gcc toolchain.
export PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-$(command -v python3)}
export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export TF_NEED_CLANG=${TF_NEED_CLANG:-0}
export TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS=${CC_OPT_FLAGS:--O2}
python3 configure.py

# Upstream has no shared-library targets for the XNNPACK delegate, so append
# them to the BUILD file (guarded so a re-run on the same checkout is a no-op).
if ! grep -q 'libtensorflowlite-delegate_xnnpack' tensorflow/lite/delegates/xnnpack/BUILD; then
  cat >> tensorflow/lite/delegates/xnnpack/BUILD <<'EOF'

cc_binary(
    name = "libtensorflowlite-delegate_xnnpack.so",
    linkopts = ["-Wl,-soname,libtensorflowlite-delegate_xnnpack.so"],
    linkshared = True,
    deps = [":xnnpack_delegate"],
)

cc_binary(
    name = "libXNNPACK.so",
    linkopts = ["-Wl,-soname,libXNNPACK.so"],
    linkshared = True,
    deps = ["@XNNPACK//:XNNPACK"],
)
EOF
fi

bazel build -c opt \
  //tensorflow/lite/c:tensorflowlite_c \
  //tensorflow/lite/delegates/xnnpack:libtensorflowlite-delegate_xnnpack.so \
  //tensorflow/lite/delegates/xnnpack:libXNNPACK.so

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/include" "$STAGE/lib"

# Collect only the headers go-tflite actually includes, plus their transitive
# includes, computed by the C preprocessor.
grep -h '#include <tensorflow/' \
  "$GO_TFLITE_ROOT"/*.go.h \
  "$GO_TFLITE_ROOT"/delegates/xnnpack/*.go.h \
  | sed 's/<\(.*\)>/"\1"/' > "$STAGE/probe.c"
gcc -M -I. "$STAGE/probe.c" \
  | tr ' \\' '\n' | grep '^tensorflow/lite/' | sort -u \
  | while read -r f; do
      install -D -m 644 "$f" "$STAGE/include/$f"
    done
install -m 755 \
  bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so \
  bazel-bin/tensorflow/lite/delegates/xnnpack/libtensorflowlite-delegate_xnnpack.so \
  bazel-bin/tensorflow/lite/delegates/xnnpack/libXNNPACK.so \
  "$STAGE/lib/"
rm -f "$STAGE/probe.c"

mkdir -p "$OUT_DIR"
NAME=go-tflite-buildkit-${BUILDKIT_SUFFIX:-$(date +%Y%m%d)}.tar.gz
tar czf "$OUT_DIR/$NAME" -C "$STAGE" include lib
echo "created: $OUT_DIR/$NAME"
